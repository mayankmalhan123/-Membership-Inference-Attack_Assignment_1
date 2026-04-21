from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms


BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
DEFAULT_SUBMISSION = BASE / "submission.csv"

MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]


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


class ResNet18WithFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 9)
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.model.fc(features)
        return logits, features


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(32),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def load_dataset(path: Path):
    dataset = torch.load(path, weights_only=False)
    dataset.transform = get_transform()
    return dataset


def load_target_model() -> ResNet18WithFeatures:
    model = ResNet18WithFeatures()
    model.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def collate_membership_batch(batch):
    sample_ids, images, labels, membership = zip(*batch)
    return (
        list(sample_ids),
        torch.stack(images),
        torch.tensor(labels, dtype=torch.long),
        list(membership),
    )


def extract_attack_features(dataset, model, batch_size: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_membership_batch,
    )

    ids = []
    labels_all = []
    membership_all = []
    blocks = []

    with torch.no_grad():
        for batch in loader:
            sample_ids, images, labels, membership = batch
            logits, _ = model(images)
            probs = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)
            batch_index = torch.arange(labels.shape[0])

            true_prob = probs[batch_index, labels][:, None]
            loss = (-log_probs[batch_index, labels])[:, None]
            entropy = (-(probs * log_probs).sum(dim=1))[:, None]
            pred = probs.argmax(dim=1)
            correct = (pred == labels).float()[:, None]
            label_one_hot = nn.functional.one_hot(labels, num_classes=9).float()

            # This mixes raw model behavior with label context.
            feature_block = torch.cat(
                [logits, probs, true_prob, loss, entropy, correct, label_one_hot],
                dim=1,
            )

            ids.extend(int(x) for x in sample_ids)
            labels_all.append(labels.cpu())
            blocks.append(feature_block.cpu())
            membership_all.extend(membership)

    features = torch.cat(blocks).float()
    labels = torch.cat(labels_all).long()
    return ids, labels, membership_all, features


def stratified_split(labels, membership, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    membership_np = np.asarray(membership, dtype=np.int64)

    train_indices = []
    val_indices = []

    for class_id in np.unique(labels_np):
        for member_value in [0, 1]:
            idx = np.where(
                (labels_np == class_id) & (membership_np == member_value)
            )[0]
            rng.shuffle(idx)
            cut = int((1.0 - val_fraction) * len(idx))
            train_indices.extend(idx[:cut])
            val_indices.extend(idx[cut:])

    return (
        torch.tensor(train_indices, dtype=torch.long),
        torch.tensor(val_indices, dtype=torch.long),
    )


def standardize(train_x, other_x: Iterable[torch.Tensor]):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_scaled = (train_x - mean) / std
    others_scaled = [((x - mean) / std) for x in other_x]
    return train_scaled, others_scaled, mean, std


class AttackMLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


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
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def roc_auc(scores, membership):
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores)) + 1

    positives = membership == 1
    n_pos = positives.sum()
    n_neg = len(scores) - n_pos
    return float(
        (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def train_attack_model(
    train_x,
    train_y,
    val_x,
    val_y,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
):
    set_seed(seed)
    model = AttackMLP(train_x.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_metric = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], 256):
            idx = permutation[start : start + 256]
            batch_x = train_x[idx]
            batch_y = train_y[idx]

            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(val_x)).numpy()
        metric = tpr_at_fpr(val_scores, val_y.numpy())

        if metric > best_metric:
            best_metric = metric
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model


def fit_on_full_public(features, membership, epochs, learning_rate, weight_decay, seed):
    set_seed(seed)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    features = (features - mean) / std
    labels = torch.tensor(membership, dtype=torch.float32)

    model = AttackMLP(features.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        permutation = torch.randperm(features.shape[0])
        model.train()
        for start in range(0, features.shape[0], 256):
            idx = permutation[start : start + 256]
            batch_x = features[idx]
            batch_y = labels[idx]
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model, mean, std


def write_submission(path: Path, sample_ids, scores):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "score"])
        for sample_id, score in zip(sample_ids, scores):
            writer.writerow([sample_id, float(score)])


def main():
    parser = argparse.ArgumentParser(
        description="Learning-first baseline for the TML 2026 membership inference task."
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "submit", "both"],
        default="both",
        help="Use a public hold-out split, generate private scores, or do both.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION)
    args = parser.parse_args()

    set_seed(args.seed)
    print("Loading public dataset, private dataset, and target model...")
    pub_ds = load_dataset(PUB_PATH)
    priv_ds = load_dataset(PRIV_PATH)
    target_model = load_target_model()

    print("Extracting features from the target model...")
    pub_ids, pub_labels, pub_membership, pub_features = extract_attack_features(
        pub_ds, target_model, args.batch_size
    )
    priv_ids, _, _, priv_features = extract_attack_features(
        priv_ds, target_model, args.batch_size
    )

    if args.mode in {"eval", "both"}:
        train_idx, val_idx = stratified_split(
            pub_labels, pub_membership, args.val_fraction, args.seed
        )

        train_x = pub_features[train_idx]
        val_x = pub_features[val_idx]
        train_y = torch.tensor(
            np.asarray(pub_membership, dtype=np.float32)[train_idx.numpy()]
        )
        val_y = torch.tensor(
            np.asarray(pub_membership, dtype=np.float32)[val_idx.numpy()]
        )

        train_x, [val_x], _, _ = standardize(train_x, [val_x])
        attack_model = train_attack_model(
            train_x,
            train_y,
            val_x,
            val_y,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )

        with torch.no_grad():
            val_scores = torch.sigmoid(attack_model(val_x)).numpy()

        val_membership = val_y.numpy().astype(np.int64)
        print(
            "Hold-out public TPR@5%FPR:",
            f"{tpr_at_fpr(val_scores, val_membership):.4f}",
        )
        print("Hold-out public ROC-AUC:", f"{roc_auc(val_scores, val_membership):.4f}")

    if args.mode in {"submit", "both"}:
        print("Training on all public samples and scoring private samples...")
        full_model, mean, std = fit_on_full_public(
            pub_features,
            pub_membership,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )

        with torch.no_grad():
            priv_scores = torch.sigmoid(
                full_model((priv_features - mean) / std)
            ).numpy()

        write_submission(args.submission_path, priv_ids, priv_scores)
        print("Saved submission to:", args.submission_path)


if __name__ == "__main__":
    main()
