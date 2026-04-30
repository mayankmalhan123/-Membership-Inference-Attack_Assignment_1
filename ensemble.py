"""Rank-mean ensemble of priv-score arrays produced by the individual attacks."""
import argparse
import sys
from pathlib import Path

import numpy as np

from lira_attack import (
    PUB_PATH,
    PRIV_PATH,
    TaskDataset,  # noqa: F401
    MembershipDataset,  # noqa: F401
    load_dataset,
    tpr_at_fpr,
    roc_auc,
    write_submission,
)


BASE = Path(__file__).parent
DEFAULT_CACHE_DIR = BASE / "lira_cache"
DEFAULT_SUBMISSION = BASE / "submission.csv"


def rank_uniform(arr):
    order = np.argsort(arr)
    out = np.empty_like(order, dtype=np.float64)
    out[order] = np.arange(len(arr))
    return out / max(len(arr) - 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION)
    p.add_argument("--inputs", nargs="+", default=None,
                   help="files inside cache-dir; default = all priv_scores_*.npy")
    p.add_argument("--equal-weights", action="store_true")
    p.add_argument("--manual-weights", type=float, nargs="+", default=None)
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.inputs is None:
        priv_files = sorted(args.cache_dir.glob("priv_scores_*.npy"))
        if not priv_files:
            print(f"no priv_scores_*.npy in {args.cache_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        priv_files = []
        for entry in args.inputs:
            path = Path(entry)
            if not path.is_absolute():
                path = args.cache_dir / entry
            if not path.exists():
                print(f"missing {path}", file=sys.stderr)
                sys.exit(1)
            priv_files.append(path)

    print(f"Loading {len(priv_files)} priv-score arrays:")
    for p_ in priv_files:
        print(f"  {p_.name}")

    pub_ds = load_dataset(PUB_PATH)
    priv_ds = load_dataset(PRIV_PATH)
    pub_membership = np.asarray([int(m) for m in pub_ds.membership], dtype=np.int64)
    priv_ids = [int(x) for x in priv_ds.ids]
    n_priv = len(priv_ids)

    priv_scores = []
    pub_scores = []
    method_names = []
    for path in priv_files:
        priv = np.load(path)
        if priv.shape[0] != n_priv:
            print(f"skipping {path.name}: length {priv.shape[0]} != {n_priv}", file=sys.stderr)
            continue
        priv_scores.append(priv)
        method = path.stem.replace("priv_scores_", "")
        method_names.append(method)
        pub_path = args.cache_dir / f"pub_scores_{method}.npy"
        if pub_path.exists():
            pub = np.load(pub_path)
            tpr = tpr_at_fpr(pub, pub_membership)
            auc = roc_auc(pub, pub_membership)
            pub_scores.append((pub, tpr, auc))
            print(f"  {method:>20s}  pub TPR@5%FPR={tpr:.4f}  AUC={auc:.4f}")
        else:
            pub_scores.append(None)
            print(f"  {method:>20s}  (no pub scores cached)")

    if args.manual_weights is not None:
        if len(args.manual_weights) != len(priv_scores):
            print("manual-weights count must match --inputs", file=sys.stderr)
            sys.exit(1)
        weights = np.array(args.manual_weights, dtype=np.float64)
    elif args.equal_weights:
        weights = np.ones(len(priv_scores), dtype=np.float64)
    else:
        # auto-weight by pub TPR above the 0.05 random baseline
        weights = np.array([
            max(s[1] - 0.05, 1e-3) if s is not None else 1e-3
            for s in pub_scores
        ], dtype=np.float64)
    weights = weights / weights.sum()

    print("\nWeights:")
    for name, w in zip(method_names, weights):
        print(f"  {name:>20s}  w={w:.4f}")

    priv_acc = np.zeros(n_priv, dtype=np.float64)
    pub_acc = None
    for w, priv, pub in zip(weights, priv_scores, pub_scores):
        priv_acc += w * rank_uniform(priv)
        if pub is not None:
            r = w * rank_uniform(pub[0])
            pub_acc = r if pub_acc is None else pub_acc + r

    if pub_acc is not None:
        tpr = tpr_at_fpr(pub_acc, pub_membership)
        auc = roc_auc(pub_acc, pub_membership)
        print(f"\n[ensemble] pub TPR@5%FPR={tpr:.4f}  AUC={auc:.4f}")
    else:
        print("\n[ensemble] (no pub eval available)")

    span = float(priv_acc.max() - priv_acc.min())
    submission = (priv_acc - priv_acc.min()) / max(span, 1e-12)
    write_submission(args.submission_path, priv_ids, submission)
    print(f"Saved {args.submission_path}")


if __name__ == "__main__":
    main()
