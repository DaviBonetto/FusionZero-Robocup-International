from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def _discover_victims_zip() -> Path | None:
    candidates = [
        Path("5_ai_training_data/1_images/rcj_victims.zip"),
        Path("New_AI/obr_overengineering_v1/dataset/rcj_victims.zip"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _looks_silver(image: np.ndarray) -> bool:
    if image is None or image.size == 0:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    spec_ratio = float(np.mean((s < 85) & (v > 165)))
    mean_v = float(np.mean(v))
    std_v = float(np.std(v))
    return (spec_ratio > 0.035 and mean_v > 115 and std_v > 20.0) or (spec_ratio > 0.055 and mean_v > 100)


def _bootstrap_from_zip(victims_zip: Path, dataset_root: Path, max_images: int) -> dict[str, int]:
    extracted = dataset_root.parent / "_bootstrap_rcj_victims"
    extracted.mkdir(parents=True, exist_ok=True)
    marker = extracted / ".done"
    if not marker.exists():
        with zipfile.ZipFile(victims_zip, "r") as zf:
            zf.extractall(extracted)
        marker.write_text("ok", encoding="utf-8")

    image_root = extracted / "rcj_victims" / "images" / "train"
    if not image_root.exists():
        return {"silver": 0, "no_silver": 0}

    silver_dir = dataset_root / "silver"
    no_silver_dir = dataset_root / "no_silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    no_silver_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(image_root.glob("*.jpg"))
    if max_images > 0:
        paths = paths[:max_images]

    copied = {"silver": 0, "no_silver": 0}
    for src in paths:
        image = cv2.imread(str(src))
        if image is None:
            continue
        label = "silver" if _looks_silver(image) else "no_silver"
        dst = (silver_dir if label == "silver" else no_silver_dir) / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied[label] += 1
    return copied


@dataclass(slots=True)
class Sample:
    path: Path
    label: int


def _collect_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for folder, label in (("no_silver", 0), ("silver", 1)):
        root = dataset_root / folder
        if not root.exists():
            continue
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            samples.append(Sample(path=path, label=label))
    return samples


def _split_samples(samples: Sequence[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    random.Random(seed).shuffle(samples)  # type: ignore[arg-type]
    split = int(round(len(samples) * (1.0 - val_ratio)))
    split = max(1, min(len(samples) - 1, split))
    return list(samples[:split]), list(samples[split:])


def _ensure_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required. Install torch to run this trainer.") from exc
    return torch, nn, DataLoader, Dataset


def train(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train quick silver-ball classifier and export TorchScript")
    parser.add_argument("--dataset-root", type=Path, default=Path("New_AI/obr_overengineering_v1/dataset/silver_ball"))
    parser.add_argument("--victims-zip", type=Path, default=None, help="Optional FusionZero victims zip")
    parser.add_argument("--bootstrap-max-images", type=int, default=1500)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model", type=Path, default=Path("New_AI/obr_overengineering_v1/models/silver_ball_classifier.pth"))
    parser.add_argument("--output-scripted", type=Path, default=Path("New_AI/obr_overengineering_v1/models/silver_ball_classifier.ts"))
    parser.add_argument("--metrics-out", type=Path, default=Path("New_AI/obr_overengineering_v1/models/silver_ball_metrics.json"))
    args = parser.parse_args(argv)

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    (args.dataset_root / "silver").mkdir(parents=True, exist_ok=True)
    (args.dataset_root / "no_silver").mkdir(parents=True, exist_ok=True)

    victims_zip = args.victims_zip or _discover_victims_zip()
    if victims_zip is not None and victims_zip.exists():
        copied = _bootstrap_from_zip(victims_zip, args.dataset_root, max_images=int(args.bootstrap_max_images))
        print(f"[bootstrap] copied from {victims_zip}: {copied}")
    else:
        print("[bootstrap] victims zip not found, using only existing labeled dataset")

    samples = _collect_samples(args.dataset_root)
    if len(samples) < 20:
        raise RuntimeError(
            f"Need at least 20 labeled images to train. Found {len(samples)} in {args.dataset_root}."
        )

    torch, nn, DataLoader, Dataset = _ensure_torch()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    train_samples, val_samples = _split_samples(samples, val_ratio=float(args.val_ratio), seed=int(args.seed))

    class SilverDataset(Dataset):  # type: ignore[misc,valid-type]
        def __init__(self, items: Sequence[Sample], size: int) -> None:
            self.items = list(items)
            self.size = int(size)

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            item = self.items[idx]
            image = cv2.imread(str(item.path))
            if image is None:
                image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
            label = torch.tensor(float(item.label), dtype=torch.float32)
            return tensor, label

    train_loader = DataLoader(SilverDataset(train_samples, args.image_size), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SilverDataset(val_samples, args.image_size), batch_size=args.batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 48, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(48, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    criterion = nn.BCEWithLogitsLoss()

    def evaluate(loader):
        model.eval()
        total = 0
        correct = 0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images).squeeze(1)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                total += int(labels.numel())
                correct += int((preds == labels).sum().item())
                tp += int(((preds == 1) & (labels == 1)).sum().item())
                fp += int(((preds == 1) & (labels == 0)).sum().item())
                fn += int(((preds == 0) & (labels == 1)).sum().item())
                tn += int(((preds == 0) & (labels == 0)).sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        acc = correct / max(1, total)
        return {"accuracy": acc, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch = int(labels.numel())
            running += float(loss.item()) * batch
            seen += batch
        train_loss = running / max(1, seen)
        val_metrics = evaluate(val_loader)
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), **{k: float(v) for k, v in val_metrics.items()}})
        print(
            f"[epoch {epoch:02d}] loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_prec={val_metrics['precision']:.3f} val_rec={val_metrics['recall']:.3f}"
        )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_model)

    scripted = torch.jit.trace(model.eval(), torch.randn(1, 3, int(args.image_size), int(args.image_size)))
    scripted.save(str(args.output_scripted))

    final_val = evaluate(val_loader)
    payload = {
        "dataset_root": str(args.dataset_root),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "image_size": int(args.image_size),
        "epochs": int(args.epochs),
        "history": history,
        "final_val": final_val,
        "output_model": str(args.output_model),
        "output_scripted": str(args.output_scripted),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] model={args.output_model} scripted={args.output_scripted} metrics={args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
