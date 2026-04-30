"""Reference-model attack: target_logit_conf - reference_logit_conf per sample."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from lira_attack import (
    PUB_PATH,
    PRIV_PATH,
    DEVICE,
    TaskDataset,  # noqa: F401
    MembershipDataset,  # noqa: F401
    build_resnet,
    load_dataset,
    load_target_model,
    stack_dataset_tensors,
    query_logit_conf,
    train_shadow,
    tpr_at_fpr,
    roc_auc,
    write_submission,
)


BASE = Path(__file__).parent
DEFAULT_REFERENCE_DIR = BASE / "reference"
DEFAULT_CACHE_DIR = BASE / "lira_cache"
DEFAULT_SUBMISSION = BASE / "submission.csv"


def rank_uniform(arr):
    order = np.argsort(arr)
    out = np.empty_like(order, dtype=np.float64)
    out[order] = np.arange(len(arr))
    return out / max(len(arr) - 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-aug", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug-train", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--blend-rmia", action="store_true")
    args = p.parse_args()

    print(f"[device] {DEVICE}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading datasets...")
    pub_ds = load_dataset(PUB_PATH)
    priv_ds = load_dataset(PRIV_PATH)
    pub_images, pub_labels = stack_dataset_tensors(pub_ds)
    priv_images, priv_labels = stack_dataset_tensors(priv_ds)
    pub_membership = np.asarray([int(m) for m in pub_ds.membership], dtype=np.int64)
    priv_ids = [int(x) for x in priv_ds.ids]
    print(f"  pub: {pub_images.shape}  priv: {priv_images.shape}")
    print(f"  members in pub: {int(pub_membership.sum())}/{len(pub_membership)}")

    args.reference_dir.mkdir(parents=True, exist_ok=True)
    ref_path = args.reference_dir / "reference.pt"
    if ref_path.exists() and not args.retrain:
        print(f"  loading cached reference from {ref_path}")
        reference = build_resnet().to(DEVICE)
        reference.load_state_dict(torch.load(ref_path, map_location=DEVICE))
        reference.eval()
    else:
        non_member_idx = np.where(pub_membership == 0)[0]
        print(f"\nTraining reference on {len(non_member_idx)} pub non-members "
              f"(epochs={args.epochs}, aug={args.aug_train})")
        t0 = time.time()
        reference = train_shadow(
            pub_images, pub_labels, non_member_idx,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            augment=args.aug_train,
        )
        print(f"  done in {time.time() - t0:.1f}s")
        torch.save(reference.state_dict(), ref_path)

    print("\nQuerying reference + target with augmentations...")
    t0 = time.time()
    ref_conf_pub = query_logit_conf(
        reference, pub_images, pub_labels, args.n_aug, args.batch_size, args.seed + 11
    )
    ref_conf_priv = query_logit_conf(
        reference, priv_images, priv_labels, args.n_aug, args.batch_size, args.seed + 12
    )
    del reference

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    target_pub_path = args.cache_dir / "target_conf_pub.npy"
    target_priv_path = args.cache_dir / "target_conf_priv.npy"
    if target_pub_path.exists() and target_priv_path.exists():
        target_conf_pub = np.load(target_pub_path).astype(np.float64)
        target_conf_priv = np.load(target_priv_path).astype(np.float64)
        print(f"  reused cached target queries from {args.cache_dir}")
    else:
        target_model = load_target_model()
        target_conf_pub = query_logit_conf(
            target_model, pub_images, pub_labels, args.n_aug, args.batch_size, args.seed
        ).astype(np.float64)
        target_conf_priv = query_logit_conf(
            target_model, priv_images, priv_labels, args.n_aug, args.batch_size, args.seed + 1
        ).astype(np.float64)
        del target_model
        np.save(target_pub_path, target_conf_pub.astype(np.float32))
        np.save(target_priv_path, target_conf_priv.astype(np.float32))
    print(f"  query phase done in {time.time() - t0:.1f}s")

    gap_pub = target_conf_pub - ref_conf_pub
    gap_priv = target_conf_priv - ref_conf_priv

    tpr = tpr_at_fpr(gap_pub, pub_membership)
    auc = roc_auc(gap_pub, pub_membership)
    print(f"\nref gap  pub TPR@5%FPR={tpr:.4f}  AUC={auc:.4f}")

    final_pub, final_priv, label = gap_pub, gap_priv, "ref_gap"

    if args.blend_rmia:
        pub_shadow_path = args.cache_dir / "shadow_conf_pub.npy"
        priv_shadow_path = args.cache_dir / "shadow_conf_priv.npy"
        membership_path = BASE / "shadows" / "membership.npy"
        if pub_shadow_path.exists() and priv_shadow_path.exists() and membership_path.exists():
            pub_shadow = np.load(pub_shadow_path)
            priv_shadow = np.load(priv_shadow_path)
            membership = np.load(membership_path)
            n_pub = pub_shadow.shape[1]
            out_mean_pub = np.empty(n_pub)
            for j in range(n_pub):
                m = membership[:, j] == 0
                out_mean_pub[j] = pub_shadow[m, j].mean() if m.any() else pub_shadow[:, j].mean()
            log_ratio_pub = target_conf_pub - out_mean_pub
            log_ratio_priv = target_conf_priv - priv_shadow.mean(axis=0)
            blend_pub = 0.5 * rank_uniform(gap_pub) + 0.5 * rank_uniform(log_ratio_pub)
            blend_priv = 0.5 * rank_uniform(gap_priv) + 0.5 * rank_uniform(log_ratio_priv)
            tpr_b = tpr_at_fpr(blend_pub, pub_membership)
            auc_b = roc_auc(blend_pub, pub_membership)
            print(f"blend (ref_gap + rmia)  pub TPR@5%FPR={tpr_b:.4f}  AUC={auc_b:.4f}")
            if tpr_b > tpr:
                final_pub, final_priv, label = blend_pub, blend_priv, "blend_ref_rmia"
        else:
            print("blend-rmia skipped: shadow cache missing")

    print(f"\nfinal: {label}  pub TPR@5%FPR={tpr_at_fpr(final_pub, pub_membership):.4f}")

    np.save(args.cache_dir / "priv_scores_reference.npy", final_priv)
    np.save(args.cache_dir / "pub_scores_reference.npy", final_pub)

    submission = final_priv.astype(np.float64)
    span = float(submission.max() - submission.min())
    submission = (submission - submission.min()) / max(span, 1e-12)
    write_submission(args.submission_path, priv_ids, submission)
    print(f"Saved {args.submission_path}")


if __name__ == "__main__":
    main()
