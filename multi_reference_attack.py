"""K reference models on disjoint random subsamples; average their gap signals."""
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
    train_shadow,
    query_logit_conf,
    tpr_at_fpr,
    roc_auc,
    write_submission,
)


BASE = Path(__file__).parent
DEFAULT_REFERENCE_DIR = BASE / "multi_reference"
DEFAULT_CACHE_DIR = BASE / "lira_cache"
DEFAULT_SUBMISSION = BASE / "submission.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION)
    p.add_argument("--n-refs", type=int, default=6)
    p.add_argument("--subsample-frac", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-aug", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug-train", action="store_true")
    p.add_argument("--retrain", action="store_true")
    args = p.parse_args()

    print(f"[device] {DEVICE}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pub_ds = load_dataset(PUB_PATH)
    priv_ds = load_dataset(PRIV_PATH)
    pub_images, pub_labels = stack_dataset_tensors(pub_ds)
    priv_images, priv_labels = stack_dataset_tensors(priv_ds)
    pub_membership = np.asarray([int(m) for m in pub_ds.membership], dtype=np.int64)
    priv_ids = [int(x) for x in priv_ds.ids]
    n_pub = pub_images.shape[0]
    print(f"  pub: {pub_images.shape}  priv: {priv_images.shape}")

    args.reference_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    non_member_idx = np.where(pub_membership == 0)[0]
    n_per_ref = int(args.subsample_frac * len(non_member_idx))
    print(f"\ntraining {args.n_refs} references; "
          f"each on {n_per_ref} of {len(non_member_idx)} pub non-members.")

    pub_conf_all = np.zeros((args.n_refs, n_pub), dtype=np.float32)
    priv_conf_all = np.zeros((args.n_refs, priv_images.shape[0]), dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    for k in range(args.n_refs):
        ckpt = args.reference_dir / f"ref_{k:02d}.pt"
        if ckpt.exists() and not args.retrain:
            print(f"\n[ref {k}] loading {ckpt}")
            model = build_resnet().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model.eval()
        else:
            subset = rng.choice(non_member_idx, size=n_per_ref, replace=False)
            print(f"\n[ref {k}] training on {len(subset)} non-members (aug={args.aug_train})")
            t0 = time.time()
            model = train_shadow(
                pub_images, pub_labels, subset,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay,
                seed=args.seed + 100 + k, augment=args.aug_train,
            )
            torch.save(model.state_dict(), ckpt)
            print(f"  done in {time.time() - t0:.1f}s")

        t0 = time.time()
        pub_conf_all[k] = query_logit_conf(
            model, pub_images, pub_labels, args.n_aug, args.batch_size, args.seed + 1000 + k
        )
        priv_conf_all[k] = query_logit_conf(
            model, priv_images, priv_labels, args.n_aug, args.batch_size, args.seed + 2000 + k
        )
        print(f"  query done in {time.time() - t0:.1f}s")
        np.save(args.cache_dir / f"priv_scores_ref{k:02d}.npy", priv_conf_all[k])
        del model

    np.save(args.cache_dir / "multi_ref_pub_conf.npy", pub_conf_all)
    np.save(args.cache_dir / "multi_ref_priv_conf.npy", priv_conf_all)

    target_pub_path = args.cache_dir / "target_conf_pub.npy"
    target_priv_path = args.cache_dir / "target_conf_priv.npy"
    if target_pub_path.exists() and target_priv_path.exists():
        target_conf_pub = np.load(target_pub_path).astype(np.float64)
        target_conf_priv = np.load(target_priv_path).astype(np.float64)
        print(f"\nreused cached target queries")
    else:
        print("\nquerying target...")
        target_model = load_target_model()
        t0 = time.time()
        target_conf_pub = query_logit_conf(
            target_model, pub_images, pub_labels, args.n_aug, args.batch_size, args.seed
        ).astype(np.float64)
        target_conf_priv = query_logit_conf(
            target_model, priv_images, priv_labels, args.n_aug, args.batch_size, args.seed + 1
        ).astype(np.float64)
        del target_model
        print(f"  done in {time.time() - t0:.1f}s")
        np.save(target_pub_path, target_conf_pub.astype(np.float32))
        np.save(target_priv_path, target_conf_priv.astype(np.float32))

    pub_score = target_conf_pub - pub_conf_all.mean(axis=0)
    priv_score = target_conf_priv - priv_conf_all.mean(axis=0)

    print("\nper-reference and average pub TPR@5%FPR:")
    for k in range(args.n_refs):
        gap = target_conf_pub - pub_conf_all[k]
        print(f"  ref {k}: TPR={tpr_at_fpr(gap, pub_membership):.4f}  AUC={roc_auc(gap, pub_membership):.4f}")
    tpr = tpr_at_fpr(pub_score, pub_membership)
    auc = roc_auc(pub_score, pub_membership)
    print(f"  AVG ({args.n_refs} refs): TPR@5%FPR={tpr:.4f}  AUC={auc:.4f}")

    np.save(args.cache_dir / "priv_scores_multi_reference.npy", priv_score)
    np.save(args.cache_dir / "pub_scores_multi_reference.npy", pub_score)

    submission = priv_score.astype(np.float64)
    span = float(submission.max() - submission.min())
    submission = (submission - submission.min()) / max(span, 1e-12)
    write_submission(args.submission_path, priv_ids, submission)
    print(f"Saved {args.submission_path}")


if __name__ == "__main__":
    main()
