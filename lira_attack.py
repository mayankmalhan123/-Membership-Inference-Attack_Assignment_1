"""LiRA shadow training, query, scoring, plus an XGBoost stacked variant."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms


BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
DEFAULT_SHADOW_DIR = BASE / "shadows"
DEFAULT_CACHE_DIR = BASE / "lira_cache"
DEFAULT_SUBMISSION = BASE / "submission.csv"

MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]
NUM_CLASSES = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        sample_id = self.ids[index]
        image = self.imgs[index]
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return sample_id, image, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        sample_id, image, label = super().__getitem__(index)
        return sample_id, image, label, self.membership[index]


def base_transform():
    return transforms.Compose(
        [transforms.Resize(32), transforms.Normalize(mean=MEAN, std=STD)]
    )


def load_dataset(path):
    dataset = torch.load(path, weights_only=False)
    dataset.transform = base_transform()
    return dataset


def build_resnet():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model


def load_target_model():
    model = build_resnet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model.to(DEVICE)
    return model


def collate(batch):
    if len(batch[0]) == 4:
        sample_ids, images, labels, membership = zip(*batch)
        return (
            list(sample_ids),
            torch.stack(images),
            torch.tensor(labels, dtype=torch.long),
            list(membership),
        )
    sample_ids, images, labels = zip(*batch)
    return list(sample_ids), torch.stack(images), torch.tensor(labels, dtype=torch.long), None


def stack_dataset_tensors(dataset):
    images = []
    labels = []
    for i in range(len(dataset)):
        out = dataset[i]
        images.append(out[1])
        labels.append(int(out[2]))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def _augment_batch(xb, pad=4):
    n, c, h, w = xb.shape
    xb_p = nn.functional.pad(xb, (pad, pad, pad, pad), mode="reflect")
    h_off = torch.randint(0, 2 * pad + 1, (n,), device=xb.device)
    w_off = torch.randint(0, 2 * pad + 1, (n,), device=xb.device)
    out = torch.empty_like(xb)
    for i in range(n):
        out[i] = xb_p[i, :, h_off[i] : h_off[i] + h, w_off[i] : w_off[i] + w]
    flip_mask = torch.rand(n, device=xb.device) < 0.5
    out[flip_mask] = torch.flip(out[flip_mask], dims=[-1])
    return out


def train_shadow(images, labels, member_idx, epochs, batch_size, lr,
                 weight_decay, seed, augment=False):
    torch.manual_seed(seed)
    model = build_resnet().to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    member_idx_t = torch.as_tensor(member_idx, dtype=torch.long)
    n = member_idx_t.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = member_idx_t[torch.randperm(n)]
        last_acc = 0
        last_total = 0
        for start in range(0, n, batch_size):
            sel = perm[start : start + batch_size]
            xb = images[sel].to(DEVICE, non_blocking=True)
            yb = labels[sel].to(DEVICE, non_blocking=True)
            if augment:
                xb = _augment_batch(xb)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_acc += (logits.argmax(1) == yb).sum().item()
            last_total += yb.shape[0]
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch + 1}/{epochs}  train_acc={last_acc / last_total:.4f}")
    model.eval()
    return model


@torch.no_grad()
def query_logit_conf(model, images, labels, n_aug, batch_size, seed):
    model.eval()
    n = images.shape[0]
    rng = np.random.default_rng(seed)
    accum = torch.zeros(n, dtype=torch.float64, device="cpu")

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
                xb_p = nn.functional.pad(xb, (pad, pad, pad, pad), mode="reflect")
                xb = xb_p[..., top : top + 32, left : left + 32]
            if flip:
                xb = torch.flip(xb, dims=[-1])

            logits = model(xb)
            log_probs = logits.log_softmax(dim=1)
            idx = torch.arange(yb.shape[0], device=yb.device)
            true_lp = log_probs[idx, yb]
            log_one_minus = torch.log1p(-true_lp.exp().clamp(max=1 - 1e-7))
            logit_conf = true_lp - log_one_minus
            accum[start : start + xb.shape[0]] += logit_conf.cpu().double()
    accum /= n_aug
    return accum.numpy()


def tpr_at_fpr(scores, membership, target_fpr=0.05):
    order = np.argsort(-scores)
    y_sorted = membership[order]
    positives = y_sorted == 1
    negatives = ~positives
    tp = np.cumsum(positives)
    fp = np.cumsum(negatives)
    tpr = tp / max(positives.sum(), 1)
    fpr = fp / max(negatives.sum(), 1)
    valid = np.where(fpr <= target_fpr)[0]
    return float(tpr[valid[-1]]) if len(valid) else 0.0


def roc_auc(scores, membership):
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores)) + 1
    pos = membership == 1
    n_pos = int(pos.sum())
    n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def write_submission(path, sample_ids, scores):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "score"])
        for sample_id, score in zip(sample_ids, scores):
            writer.writerow([sample_id, float(score)])


def phase_train(args, pub_images, pub_labels, n_pub, shadow_dir):
    shadow_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    membership = np.zeros((args.n_shadows, n_pub), dtype=np.uint8)
    for k in range(args.n_shadows):
        idx = rng.permutation(n_pub)
        cut = int(args.shadow_fraction * n_pub)
        membership[k, idx[:cut]] = 1

    np.save(shadow_dir / "membership.npy", membership)
    with open(shadow_dir / "config.json", "w") as f:
        json.dump(
            {
                "n_shadows": args.n_shadows,
                "shadow_fraction": args.shadow_fraction,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            },
            f,
        )

    for k in range(args.n_shadows):
        ckpt_path = shadow_dir / f"shadow_{k:02d}.pt"
        if ckpt_path.exists() and not args.retrain:
            print(f"[shadow {k}] cached, skipping")
            continue
        member_idx = np.where(membership[k] == 1)[0]
        t0 = time.time()
        print(f"[shadow {k}] training on {len(member_idx)} samples...")
        model = train_shadow(
            pub_images,
            pub_labels,
            member_idx,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + k,
            augment=args.aug_shadows,
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"[shadow {k}] done in {time.time() - t0:.1f}s")
    return membership


def phase_query(args, pub_images, pub_labels, priv_images, priv_labels,
                shadow_dir, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    pub_conf_path = cache_dir / "shadow_conf_pub.npy"
    priv_conf_path = cache_dir / "shadow_conf_priv.npy"
    if pub_conf_path.exists() and priv_conf_path.exists() and not args.requery:
        print(f"[query] using cached confidences from {cache_dir}")
        return np.load(pub_conf_path), np.load(priv_conf_path)

    pub_conf = np.zeros((args.n_shadows, pub_images.shape[0]), dtype=np.float32)
    priv_conf = np.zeros((args.n_shadows, priv_images.shape[0]), dtype=np.float32)

    for k in range(args.n_shadows):
        ckpt_path = shadow_dir / f"shadow_{k:02d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing shadow checkpoint: {ckpt_path}")
        model = build_resnet().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        t0 = time.time()
        pub_conf[k] = query_logit_conf(
            model, pub_images, pub_labels, args.n_aug, args.batch_size, args.seed + k
        )
        priv_conf[k] = query_logit_conf(
            model, priv_images, priv_labels, args.n_aug, args.batch_size, args.seed + 1000 + k
        )
        print(f"[query {k}] {time.time() - t0:.1f}s")
        del model

    np.save(pub_conf_path, pub_conf)
    np.save(priv_conf_path, priv_conf)
    return pub_conf, priv_conf


def per_sample_lira_pub(pub_conf, membership, target_conf):
    # online LiRA: per-sample IN/OUT Gaussians, score = log p_in - log p_out
    n_pub = pub_conf.shape[1]
    scores = np.zeros(n_pub, dtype=np.float64)
    fixed_var = 1e-2

    for j in range(n_pub):
        in_mask = membership[:, j] == 1
        out_mask = ~in_mask
        in_vals = pub_conf[in_mask, j]
        out_vals = pub_conf[out_mask, j]
        if in_vals.size < 2 or out_vals.size < 2:
            scores[j] = target_conf[j]
            continue
        mu_in = in_vals.mean()
        var_in = max(float(in_vals.var(ddof=1)), fixed_var)
        mu_out = out_vals.mean()
        var_out = max(float(out_vals.var(ddof=1)), fixed_var)
        t = float(target_conf[j])
        log_p_in = -0.5 * ((t - mu_in) ** 2 / var_in + math.log(2 * math.pi * var_in))
        log_p_out = -0.5 * ((t - mu_out) ** 2 / var_out + math.log(2 * math.pi * var_out))
        scores[j] = log_p_in - log_p_out
    return scores


def offline_lira_priv(priv_conf, target_conf_priv, pub_conf, membership,
                      pub_labels, priv_labels):
    # priv samples have no shadow IN-set; pool an IN Gaussian per class from pub
    fixed_var = 1e-2
    n_classes = NUM_CLASSES
    in_pooled = [[] for _ in range(n_classes)]
    for j in range(pub_conf.shape[1]):
        in_mask = membership[:, j] == 1
        if in_mask.sum() == 0:
            continue
        c = int(pub_labels[j])
        in_pooled[c].append(pub_conf[in_mask, j])
    in_stats = []
    for c in range(n_classes):
        if in_pooled[c]:
            arr = np.concatenate(in_pooled[c])
            in_stats.append((arr.mean(), max(float(arr.var(ddof=1)), fixed_var)))
        else:
            in_stats.append((0.0, 1.0))

    n_priv = priv_conf.shape[1]
    scores = np.zeros(n_priv, dtype=np.float64)
    for j in range(n_priv):
        out_vals = priv_conf[:, j]
        mu_out = float(out_vals.mean())
        var_out = max(float(out_vals.var(ddof=1)), fixed_var)
        c = int(priv_labels[j])
        mu_in, var_in = in_stats[c]
        t = float(target_conf_priv[j])
        log_p_in = -0.5 * ((t - mu_in) ** 2 / var_in + math.log(2 * math.pi * var_in))
        log_p_out = -0.5 * ((t - mu_out) ** 2 / var_out + math.log(2 * math.pi * var_out))
        scores[j] = log_p_in - log_p_out
    return scores


def stack_with_xgb(pub_lira, priv_lira, pub_conf, priv_conf,
                   pub_membership_label, pub_labels,
                   target_conf_pub, target_conf_priv, n_splits, seed):
    import xgboost as xgb

    def feat(lira, sconf, tconf, labels):
        sconf = sconf.T  # (N, K)
        feats = [
            lira[:, None],
            tconf[:, None],
            sconf.mean(axis=1, keepdims=True),
            sconf.std(axis=1, keepdims=True),
            sconf.min(axis=1, keepdims=True),
            sconf.max(axis=1, keepdims=True),
            np.median(sconf, axis=1, keepdims=True),
            (tconf[:, None] - sconf.mean(axis=1, keepdims=True)),
            (tconf[:, None] - sconf.mean(axis=1, keepdims=True))
            / (sconf.std(axis=1, keepdims=True) + 1e-6),
            np.eye(NUM_CLASSES, dtype=np.float32)[labels],
        ]
        return np.concatenate(feats, axis=1).astype(np.float32)

    X_pub = feat(pub_lira, pub_conf, target_conf_pub, pub_labels)
    X_priv = feat(priv_lira, priv_conf, target_conf_priv,
                  np.array([0] * priv_conf.shape[1], dtype=np.int64))

    rng = np.random.default_rng(seed)
    n = X_pub.shape[0]
    fold = np.full(n, -1, dtype=np.int64)
    for class_id in range(NUM_CLASSES):
        for member_value in [0, 1]:
            mask = (pub_labels == class_id) & (pub_membership_label == member_value)
            idx = np.where(mask)[0]
            rng.shuffle(idx)
            for k, sample in enumerate(idx):
                fold[sample] = k % n_splits

    oof = np.zeros(n, dtype=np.float64)
    test = np.zeros(X_priv.shape[0], dtype=np.float64)
    for k in range(n_splits):
        train_mask = fold != k
        val_mask = fold == k
        model = xgb.XGBClassifier(
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            early_stopping_rounds=50,
            random_state=seed + k,
            n_jobs=-1,
        )
        model.fit(
            X_pub[train_mask],
            pub_membership_label[train_mask],
            eval_set=[(X_pub[val_mask], pub_membership_label[val_mask])],
            verbose=False,
        )
        oof[val_mask] = model.predict_proba(X_pub[val_mask])[:, 1]
        test += model.predict_proba(X_priv)[:, 1] / n_splits

    return oof, test


def main():
    parser = argparse.ArgumentParser(
        description="LiRA membership inference attack with optional XGBoost stacking."
    )
    parser.add_argument("--phase", choices=["train", "score", "all"], default="all")
    parser.add_argument("--n-shadows", type=int, default=16)
    parser.add_argument("--shadow-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-aug", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--requery", action="store_true")
    parser.add_argument("--no-stack", action="store_true")
    parser.add_argument("--aug-shadows", action="store_true",
                        help="train shadows with crop+flip aug")
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION)
    args = parser.parse_args()

    print(f"[device] {DEVICE}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading datasets...")
    pub_ds = load_dataset(PUB_PATH)
    priv_ds = load_dataset(PRIV_PATH)
    pub_images, pub_labels = stack_dataset_tensors(pub_ds)
    priv_images, priv_labels = stack_dataset_tensors(priv_ds)
    n_pub = pub_images.shape[0]
    pub_membership_label = np.asarray(
        [int(m) for m in pub_ds.membership], dtype=np.int64
    )
    pub_ids = [int(x) for x in pub_ds.ids]
    priv_ids = [int(x) for x in priv_ds.ids]
    print(f"  pub: {pub_images.shape}, priv: {priv_images.shape}")

    if args.phase in {"train", "all"}:
        membership = phase_train(args, pub_images, pub_labels, n_pub, args.shadow_dir)
    else:
        membership = np.load(args.shadow_dir / "membership.npy")

    if args.phase == "train":
        return

    print("\n[query] target model on pub + priv with augmentation...")
    target_model = load_target_model()
    target_conf_pub = query_logit_conf(
        target_model, pub_images, pub_labels, args.n_aug, args.batch_size, args.seed
    )
    target_conf_priv = query_logit_conf(
        target_model, priv_images, priv_labels, args.n_aug, args.batch_size, args.seed + 1
    )
    del target_model

    pub_conf, priv_conf = phase_query(
        args, pub_images, pub_labels, priv_images, priv_labels,
        args.shadow_dir, args.cache_dir
    )

    print("\n[lira] computing per-sample LiRA scores...")
    pub_lira = per_sample_lira_pub(pub_conf, membership, target_conf_pub)
    priv_lira = offline_lira_priv(
        priv_conf, target_conf_priv, pub_conf, membership,
        pub_labels.numpy(), priv_labels.numpy()
    )

    pub_lira_metric = tpr_at_fpr(pub_lira, pub_membership_label)
    pub_lira_auc = roc_auc(pub_lira, pub_membership_label)
    print(f"  raw LiRA on pub: TPR@5%FPR = {pub_lira_metric:.4f}  AUC = {pub_lira_auc:.4f}")

    if args.no_stack:
        scores = priv_lira
        score_name = "raw LiRA"
    else:
        print("\n[stack] XGBoost on LiRA + shadow-stats + target features...")
        stack_oof, stack_test = stack_with_xgb(
            pub_lira, priv_lira, pub_conf, priv_conf, pub_membership_label,
            pub_labels.numpy(), target_conf_pub, target_conf_priv,
            n_splits=args.n_splits, seed=args.seed,
        )
        oof_metric = tpr_at_fpr(stack_oof, pub_membership_label)
        oof_auc = roc_auc(stack_oof, pub_membership_label)
        print(f"  stacked OOF: TPR@5%FPR = {oof_metric:.4f}  AUC = {oof_auc:.4f}")
        scores = stack_test
        score_name = "stacked"

    scores = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    write_submission(args.submission_path, priv_ids, scores)
    print(f"\nSaved submission ({score_name}) to: {args.submission_path}")


if __name__ == "__main__":
    main()
