"""
Fine-tune MobileNetV3-Small for chart / non-chart binary classification.

Positive : raw_data/charts/
Negative : raw_data/not_charts/raw_data/coco_val/val2017/ (all images)

Usage:
    python train.py
    python train.py --epochs 10
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

CHARTS_DIR = Path(__file__).parent.parent / "raw_data" / "charts"
NEG_DIR    = Path(__file__).parent.parent / "raw_data" / "not_charts" / "raw_data" / "coco_val" / "val2017"
MODEL_DIR  = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "chart_classifier.pth"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ChartDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        src, label = self.items[idx]
        img = Image.open(src).convert("RGB") if isinstance(src, (str, Path)) else src.convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def build_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for param in model.features[:9].parameters():
        param.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def load_positives():
    items = []
    for folder in sorted(CHARTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        jpgs = list(folder.glob("*.jpg"))
        items.extend(jpgs)
        print(f"  {folder.name}: {len(jpgs):,} images")
    return items


def load_negatives():
    files = list(NEG_DIR.rglob("*.jpg")) + list(NEG_DIR.rglob("*.jpeg")) + list(NEG_DIR.rglob("*.png"))
    print(f"  Loaded {len(files):,} negatives from {NEG_DIR}")
    return files


def run(args):
    MODEL_DIR.mkdir(exist_ok=True)
    random.seed(42)

    print("\n[1/4] Positive samples")
    pos_paths = load_positives()
    print(f"  Total: {len(pos_paths):,}\n")

    print("[2/4] Negative samples")
    neg_paths = load_negatives()

    # balance: subsample whichever side is larger
    n = min(len(pos_paths), len(neg_paths))
    pos_paths = random.sample(pos_paths, n)
    neg_paths = random.sample(neg_paths, n)
    print(f"  Balanced: {n:,} pos / {n:,} neg\n")

    all_items = [(p, 1) for p in pos_paths] + [(p, 0) for p in neg_paths]
    random.shuffle(all_items)

    split = int(len(all_items) * 0.9)
    train_items, val_items = all_items[:split], all_items[split:]

    train_ds = ChartDataset(train_items, train_tf)
    val_ds   = ChartDataset(val_items,   val_tf)

    labels  = [lbl for _, lbl in train_items]
    counts  = [labels.count(0), labels.count(1)]
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,  num_workers=0)

    print(f"[3/4] Training  (train={len(train_ds):,}  val={len(val_ds):,})")
    model = build_model()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_recall = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = (torch.sigmoid(model(imgs).squeeze(1)) >= 0.35).float()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        avg_loss  = total_loss / len(train_loader)
        print(f"  epoch {epoch:02d}/{args.epochs}  loss={avg_loss:.4f}  recall={recall:.3f}  precision={precision:.3f}")

        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"    saved  (best recall={best_recall:.3f})")

    print(f"\n[4/4] Done -> {MODEL_PATH}  (best recall={best_recall:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    run(p.parse_args())
