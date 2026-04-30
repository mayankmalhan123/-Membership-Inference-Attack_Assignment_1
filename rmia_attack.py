"""RMIA (Zarifzadeh et al. 2023) on cached LiRA shadow confidences."""
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
    load_dataset,
    load_target_model,
    stack_dataset_tensors,
    query_logit_conf,
    tpr_at_fpr,
    roc_auc,
    write_submission,
)


BASE = Path(__file__).parent
DEFAULT_SHADOW_DIR = BASE / "shadows"
DEFAULT_CACHE_DIR = BASE / "lira_cache"
DEFAULT_SUBMISSION = BASE / "submission.csv"


def out_shadow_mean(shadow_conf, membership):
    if membership is None:
        return shadow_conf.mean(axis=0).astype(np.float64)
    n = shadow_conf.shape[1]
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        m = membership[:, j] == 0
        out[j] = float(shadow_conf[m, j].mean()) if m.any() else float(shadow_conf[:, j].mean())
    return out


def out_shadow_std(shadow_conf, membership, floor=1e-3):
    if membership is None:
        return np.maximum(shadow_conf.std(axis=0, ddof=1), floor).astype(np.float64)
    n = shadow_conf.shape[1]
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        m = membership[:, j] == 0
        vals = shadow_conf[m, j] if m.sum() >= 2 else shadow_conf[:, j]
        out[j] = max(float(vals.std(ddof=1)), floor)
    return out


def rmia_pairwise(log_ratio_target, log_ratio_ref, log_beta):
    # for each x: fraction of ref samples z with log_ratio(x) - log_ratio(z) > log(beta)
    if len(log_ratio_ref) == 0:
        return np.zeros_like(log_ratio_target, dtype=np.float64)
    ref_sorted = np.sort(log_ratio_ref)
    threshold = log_ratio_target - log_beta
    counts = np.searchsorted(ref_sorted, threshold, side="left")
    return counts.astype(np.float64) / len(ref_sorted)


def rank_uniform(arr):
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr))
    return ranks / max(len(arr) - 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION)
    p.add_argument("--n-aug", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--betas", type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 3.0, 5.0])
    p.add_argument("--blend-zscore", action="store_true")
    p.add_argument("--retarget", action="store_true")
    args = p.parse_args()

    print(f"[device] {DEVICE}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pub_conf_path = args.cache_dir / "shadow_conf_pub.npy"
    priv_conf_path = args.cache_dir / "shadow_conf_priv.npy"
    membership_path = args.shadow_dir / "membership.npy"
    for path in (pub_conf_path, priv_conf_path, membership_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run lira_attack.py first")

    print("Loading datasets and shadow caches...")
    pub_ds = load_dataset(PUB_PATH)
    priv_ds = load_dataset(PRIV_PATH)
    pub_images, pub_labels = stack_dataset_tensors(pub_ds)
    priv_images, priv_labels = stack_dataset_tensors(priv_ds)
    pub_membership = np.asarray([int(m) for m in pub_ds.membership], dtype=np.int64)
    priv_ids = [int(x) for x in priv_ds.ids]

    pub_shadow_conf = np.load(pub_conf_path)
    priv_shadow_conf = np.load(priv_conf_path)
    membership = np.load(membership_path)
    print(f"  pub_shadow_conf={pub_shadow_conf.shape}  priv_shadow_conf={priv_shadow_conf.shape}")

    target_pub_path = args.cache_dir / "target_conf_pub.npy"
    target_priv_path = args.cache_dir / "target_conf_priv.npy"
    if target_pub_path.exists() and target_priv_path.exists() and not args.retarget:
        target_conf_pub = np.load(target_pub_path).astype(np.float64)
        target_conf_priv = np.load(target_priv_path).astype(np.float64)
        print(f"  reused target queries from {args.cache_dir}")
    else:
        print(f"\nQuerying target with {args.n_aug} augmentation views...")
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

    out_mean_pub = out_shadow_mean(pub_shadow_conf, membership)
    out_mean_priv = out_shadow_mean(priv_shadow_conf, None)
    log_ratio_pub = target_conf_pub - out_mean_pub
    log_ratio_priv = target_conf_priv - out_mean_priv

    out_std_pub = out_shadow_std(pub_shadow_conf, membership)
    out_std_priv = out_shadow_std(priv_shadow_conf, None)
    z_pub = (target_conf_pub - out_mean_pub) / out_std_pub
    z_priv = (target_conf_priv - out_mean_priv) / out_std_priv

    print("\n  log_ratio  pub TPR@5%FPR={:.4f}  AUC={:.4f}".format(
        tpr_at_fpr(log_ratio_pub, pub_membership),
        roc_auc(log_ratio_pub, pub_membership)))
    print("  z_score    pub TPR@5%FPR={:.4f}  AUC={:.4f}".format(
        tpr_at_fpr(z_pub, pub_membership), roc_auc(z_pub, pub_membership)))

    log_ratio_ref = log_ratio_pub[pub_membership == 0]
    print(f"  reference population: {len(log_ratio_ref)} pub non-members")

    print("\nbeta sweep on pub OOF:")
    best_name = "log_ratio"
    best_tpr = tpr_at_fpr(log_ratio_pub, pub_membership)
    best_pub = log_ratio_pub
    best_priv = log_ratio_priv
    for beta in args.betas:
        log_beta = float(np.log(beta))
        rmia_pub = rmia_pairwise(log_ratio_pub, log_ratio_ref, log_beta)
        rmia_priv = rmia_pairwise(log_ratio_priv, log_ratio_ref, log_beta)
        tpr = tpr_at_fpr(rmia_pub, pub_membership)
        auc = roc_auc(rmia_pub, pub_membership)
        print(f"  beta={beta:>4.2f}  TPR@5%FPR={tpr:.4f}  AUC={auc:.4f}")
        if tpr > best_tpr:
            best_name = f"rmia_beta{beta}"
            best_tpr = tpr
            best_pub = rmia_pub
            best_priv = rmia_priv

    print(f"\nbest: {best_name}  pub TPR@5%FPR={best_tpr:.4f}")

    if args.blend_zscore:
        blend_pub = 0.5 * rank_uniform(best_pub) + 0.5 * rank_uniform(z_pub)
        blend_priv = 0.5 * rank_uniform(best_priv) + 0.5 * rank_uniform(z_priv)
        tpr_blend = tpr_at_fpr(blend_pub, pub_membership)
        auc_blend = roc_auc(blend_pub, pub_membership)
        print(f"blend (rmia+zscore): TPR@5%FPR={tpr_blend:.4f}  AUC={auc_blend:.4f}")
        if tpr_blend > best_tpr:
            best_name = f"blend({best_name}+zscore)"
            best_priv = blend_priv
            best_pub = blend_pub
            best_tpr = tpr_blend

    print(f"final: {best_name}  pub TPR@5%FPR={best_tpr:.4f}")

    np.save(args.cache_dir / "priv_scores_rmia.npy", best_priv)
    np.save(args.cache_dir / "pub_scores_rmia.npy", best_pub)

    submission = best_priv.astype(np.float64)
    span = float(submission.max() - submission.min())
    submission = (submission - submission.min()) / max(span, 1e-12)
    write_submission(args.submission_path, priv_ids, submission)
    print(f"Saved {args.submission_path}")


if __name__ == "__main__":
    main()
