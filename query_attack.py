"""Query-only attack: augmentation consistency + additive-noise robustness."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lira_attack import (
    PUB_PATH,
    PRIV_PATH,
    DEVICE,
    TaskDataset,  # noqa: F401
    MembershipDataset,  # noqa: F401
    load_dataset,
    load_target_model,
    stack_dataset_tensors,
    tpr_at_fpr,
    roc_auc,
    write_submission,
)


BASE = Path(__file__).parent
DEFAULT_CACHE_DIR = BASE / "lira_cache"
DEFAULT_SHADOW_DIR = BASE / "shadows"
DEFAULT_SUBMISSION = BASE / "submission.csv"


@torch.no_grad()
def collect_query_features(model, images, labels, n_aug, noise_levels, batch_size, seed):
    model.eval()
    n = images.shape[0]
    rng = np.random.default_rng(seed)

    logit_conf_views = np.zeros((n_aug, n), dtype=np.float32)
    margin_views = np.zeros((n_aug, n), dtype=np.float32)
    pred_views = np.zeros((n_aug, n), dtype=np.int32)

    for view in range(n_aug):
        flip = view % 2 == 1 and view > 0
        if view <= 1:
            top = left = 4
        else:
            top = int(rng.integers(0, 9))
            left = int(rng.integers(0, 9))

        for start in range(0, n, batch_size):
            xb = images[start : start + batch_size].to(DEVICE, non_blocking=True)
            yb = labels[start : start + batch_size].to(DEVICE, non_blocking=True)
            if view > 1:
                pad = 4
                xb = nn.functional.pad(xb, (pad, pad, pad, pad), mode="reflect")
                xb = xb[..., top : top + 32, left : left + 32]
            if flip:
                xb = torch.flip(xb, dims=[-1])

            logits = model(xb)
            log_probs = logits.log_softmax(dim=1)
            idx = torch.arange(yb.shape[0], device=yb.device)
            true_lp = log_probs[idx, yb]
            log_one_minus = torch.log1p(-true_lp.exp().clamp(max=1 - 1e-7))
            logit_conf = (true_lp - log_one_minus).cpu().numpy()
            top2 = torch.topk(logits, 2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            pred = logits.argmax(1).cpu().numpy()
            sl = slice(start, start + xb.shape[0])
            logit_conf_views[view, sl] = logit_conf
            margin_views[view, sl] = margin
            pred_views[view, sl] = pred

    mean_logit_conf = logit_conf_views.mean(axis=0)
    std_logit_conf = logit_conf_views.std(axis=0, ddof=1) if n_aug >= 2 else np.zeros(n)
    mean_margin = margin_views.mean(axis=0)
    pred_mode = np.apply_along_axis(lambda c: np.bincount(c).argmax(), 0, pred_views)
    pred_consistency = (pred_views == pred_mode[None, :]).mean(axis=0)

    noise_conf = np.zeros((len(noise_levels), n), dtype=np.float32)
    for ni, sigma in enumerate(noise_levels):
        torch.manual_seed(seed + 17 * ni + 1)
        for start in range(0, n, batch_size):
            xb = images[start : start + batch_size].to(DEVICE, non_blocking=True)
            yb = labels[start : start + batch_size].to(DEVICE, non_blocking=True)
            if sigma > 0.0:
                xb = xb + torch.randn_like(xb) * sigma
            logits = model(xb)
            log_probs = logits.log_softmax(dim=1)
            idx = torch.arange(yb.shape[0], device=yb.device)
            true_lp = log_probs[idx, yb]
            log_one_minus = torch.log1p(-true_lp.exp().clamp(max=1 - 1e-7))
            noise_conf[ni, start : start + xb.shape[0]] = (true_lp - log_one_minus).cpu().numpy()

    if len(noise_levels) >= 2:
        x = np.array(noise_levels, dtype=np.float32)
        x_centered = x - x.mean()
        denom = float((x_centered ** 2).sum()) or 1.0
        slope = ((noise_conf - noise_conf.mean(axis=0, keepdims=True)) * x_centered[:, None]).sum(axis=0) / denom
    else:
        slope = np.zeros(n, dtype=np.float32)

    return {
        "mean_logit_conf": mean_logit_conf,
        "neg_std_logit_conf": -std_logit_conf,
        "mean_margin": mean_margin,
        "pred_consistency": pred_consistency,
        "noise_slope": slope,
        "mean_noise_conf": noise_conf.mean(axis=0),
        "high_noise_conf": noise_conf[-1],
    }


def rank_uniform(arr):
    order = np.argsort(arr)
    out = np.empty_like(order, dtype=np.float64)
    out[order] = np.arange(len(arr))
    return out / max(len(arr) - 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    p.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION)
    p.add_argument("--n-aug", type=int, default=40)
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blend-rmia", action="store_true")
    p.add_argument("--rmia-weight", type=float, default=0.5)
    p.add_argument("--feat-cache", type=str, default="query_feats.npz")
    p.add_argument("--recompute", action="store_true")
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

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    feat_path = args.cache_dir / args.feat_cache
    if feat_path.exists() and not args.recompute:
        cache = np.load(feat_path)
        feats_pub = {k.removeprefix("pub_"): cache[k] for k in cache.files if k.startswith("pub_")}
        feats_priv = {k.removeprefix("priv_"): cache[k] for k in cache.files if k.startswith("priv_")}
    else:
        target = load_target_model()
        t0 = time.time()
        feats_pub = collect_query_features(
            target, pub_images, pub_labels, args.n_aug, args.noise_levels, args.batch_size, args.seed,
        )
        feats_priv = collect_query_features(
            target, priv_images, priv_labels, args.n_aug, args.noise_levels, args.batch_size, args.seed + 1000,
        )
        del target
        print(f"  query done in {time.time() - t0:.1f}s")
        np.savez(feat_path,
                 **{f"pub_{k}": v for k, v in feats_pub.items()},
                 **{f"priv_{k}": v for k, v in feats_priv.items()})

    print("\nper-feature signal on pub:")
    feat_aucs = {}
    for k in feats_pub:
        auc = roc_auc(feats_pub[k], pub_membership)
        tpr = tpr_at_fpr(feats_pub[k], pub_membership)
        feat_aucs[k] = auc
        flag = " *" if auc > 0.5 else ""
        print(f"  {k:>22s}  AUC={auc:.4f}  TPR@5%FPR={tpr:.4f}{flag}")

    pub_acc = np.zeros(pub_images.shape[0], dtype=np.float64)
    priv_acc = np.zeros(priv_images.shape[0], dtype=np.float64)
    weight_sum = 0.0
    for k, auc in feat_aucs.items():
        sign = 1.0 if auc >= 0.5 else -1.0
        w = abs(auc - 0.5)
        if w < 1e-3:
            continue
        pub_acc += w * rank_uniform(sign * feats_pub[k])
        priv_acc += w * rank_uniform(sign * feats_priv[k])
        weight_sum += w
    pub_combined = pub_acc / max(weight_sum, 1e-9)
    priv_combined = priv_acc / max(weight_sum, 1e-9)

    tpr_combined = tpr_at_fpr(pub_combined, pub_membership)
    auc_combined = roc_auc(pub_combined, pub_membership)
    print(f"\nrank-mean (AUC-weighted): pub TPR@5%FPR={tpr_combined:.4f}  AUC={auc_combined:.4f}")

    final_pub, final_priv, label = pub_combined, priv_combined, "query"

    if args.blend_rmia:
        pub_conf_path = args.cache_dir / "shadow_conf_pub.npy"
        priv_conf_path = args.cache_dir / "shadow_conf_priv.npy"
        target_pub_path = args.cache_dir / "target_conf_pub.npy"
        target_priv_path = args.cache_dir / "target_conf_priv.npy"
        membership_path = args.shadow_dir / "membership.npy"
        if all(pp.exists() for pp in (pub_conf_path, priv_conf_path, target_pub_path, target_priv_path, membership_path)):
            pub_shadow = np.load(pub_conf_path)
            priv_shadow = np.load(priv_conf_path)
            target_pub = np.load(target_pub_path).astype(np.float64)
            target_priv = np.load(target_priv_path).astype(np.float64)
            membership = np.load(membership_path)
            n_pub = pub_shadow.shape[1]
            out_mean = np.empty(n_pub)
            for j in range(n_pub):
                m = membership[:, j] == 0
                out_mean[j] = pub_shadow[m, j].mean() if m.any() else pub_shadow[:, j].mean()
            log_ratio_pub = target_pub - out_mean
            log_ratio_priv = target_priv - priv_shadow.mean(axis=0)
            blend_pub = (1 - args.rmia_weight) * rank_uniform(pub_combined) + args.rmia_weight * rank_uniform(log_ratio_pub)
            blend_priv = (1 - args.rmia_weight) * rank_uniform(priv_combined) + args.rmia_weight * rank_uniform(log_ratio_priv)
            tpr_b = tpr_at_fpr(blend_pub, pub_membership)
            print(f"blend (query + rmia, w_rmia={args.rmia_weight}): pub TPR@5%FPR={tpr_b:.4f}")
            if tpr_b > tpr_combined:
                final_pub, final_priv, label = blend_pub, blend_priv, f"blend_query_rmia_w{args.rmia_weight}"

    print(f"\nfinal: {label}  pub TPR@5%FPR={tpr_at_fpr(final_pub, pub_membership):.4f}")
    np.save(args.cache_dir / "priv_scores_query.npy", final_priv)
    np.save(args.cache_dir / "pub_scores_query.npy", final_pub)

    submission = final_priv.astype(np.float64)
    span = float(submission.max() - submission.min())
    submission = (submission - submission.min()) / max(span, 1e-12)
    write_submission(args.submission_path, priv_ids, submission)
    print(f"Saved {args.submission_path}")


if __name__ == "__main__":
    main()
