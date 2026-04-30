"""PGD-distance probe: count of epsilon levels where the prediction survives."""
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
DEFAULT_SUBMISSION = BASE / "submission.csv"


def pgd_untargeted(model, x, y, epsilon, alpha, n_steps):
    model.eval()
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    for _ in range(n_steps):
        delta.requires_grad_(True)
        out = model(x + delta)
        loss = nn.functional.cross_entropy(out, y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta.detach() + alpha * grad.sign()).clamp_(-epsilon, epsilon)
    with torch.no_grad():
        return model(x + delta).argmax(dim=1)


@torch.no_grad()
def clean_predictions(model, images, batch_size):
    model.eval()
    preds = torch.empty(images.shape[0], dtype=torch.long)
    for start in range(0, images.shape[0], batch_size):
        xb = images[start : start + batch_size].to(DEVICE)
        preds[start : start + xb.shape[0]] = model(xb).argmax(dim=1).cpu()
    return preds


def robustness_count(model, images, labels, epsilons, alpha_frac, n_steps, batch_size):
    n = images.shape[0]
    survived = torch.zeros(n, dtype=torch.float32)
    print(f"  clean predictions on {n} samples...")
    clean = clean_predictions(model, images, batch_size)
    for eps in epsilons:
        alpha = float(alpha_frac) * eps
        t0 = time.time()
        for start in range(0, n, batch_size):
            xb = images[start : start + batch_size].to(DEVICE)
            yb = clean[start : start + xb.shape[0]].to(DEVICE)
            adv_pred = pgd_untargeted(model, xb, yb, eps, alpha, n_steps)
            survived[start : start + xb.shape[0]] += (adv_pred == yb).float().cpu()
        print(f"    eps={eps:>5.3f}  alpha={alpha:.4f}  steps={n_steps}  in {time.time() - t0:.1f}s")
    return survived.numpy() / len(epsilons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epsilons", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--alpha-frac", type=float, default=0.25)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION)
    args = p.parse_args()

    print(f"[device] {DEVICE}")
    print(f"epsilons={args.epsilons}  alpha_frac={args.alpha_frac}  steps={args.n_steps}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pub_ds = load_dataset(PUB_PATH)
    priv_ds = load_dataset(PRIV_PATH)
    pub_images, pub_labels = stack_dataset_tensors(pub_ds)
    priv_images, priv_labels = stack_dataset_tensors(priv_ds)
    pub_membership = np.asarray([int(m) for m in pub_ds.membership], dtype=np.int64)
    priv_ids = [int(x) for x in priv_ds.ids]
    target = load_target_model()

    print("\npub:")
    t0 = time.time()
    pub_scores = robustness_count(
        target, pub_images, pub_labels,
        args.epsilons, args.alpha_frac, args.n_steps, args.batch_size,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("\npriv:")
    t0 = time.time()
    priv_scores = robustness_count(
        target, priv_images, priv_labels,
        args.epsilons, args.alpha_frac, args.n_steps, args.batch_size,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    tpr = tpr_at_fpr(pub_scores, pub_membership)
    auc = roc_auc(pub_scores, pub_membership)
    print(f"\npub TPR@5%FPR={tpr:.4f}  AUC={auc:.4f}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.cache_dir / "priv_scores_adversarial.npy", priv_scores)
    np.save(args.cache_dir / "pub_scores_adversarial.npy", pub_scores)

    submission = priv_scores.astype(np.float64)
    span = float(submission.max() - submission.min())
    submission = (submission - submission.min()) / max(span, 1e-12)
    write_submission(args.submission_path, priv_ids, submission)
    print(f"Saved {args.submission_path}")


if __name__ == "__main__":
    main()
