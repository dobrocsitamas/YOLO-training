"""
train_classifier.py
-------------------
Unified vehicle sub-category classifier (6 classes):
  light_truck | medium_truck | heavy_truck | vehicle_combination | bus_solo | bus_articulated

Bemenet:  Training_pictures_reviewed/{class_name}/*.jpg  +  *.txt (YOLO bbox)
Modell:   EfficientNet-B0 (torchvision) – fine-tune teljes hálózat
Kimenet:  vehicle_classifier.pt  (torch.save dict, tartalmazza a class neveket is)

Futtatás:
  .venv\Scripts\python scripts\train_classifier.py
  .venv\Scripts\python scripts\train_classifier.py --epochs 60 --batch 32
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ── Osztályok (sorrend rögzített!) ────────────────────────────────────────────
# medium_truck + heavy_truck összevonva → "heavy_truck"
CLASSES = [
    "light_truck",
    "heavy_truck",
    "vehicle_combination",
    "bus_solo",
    "bus_articulated",
]

# Melyik mappa melyik osztályba kerül (több mappa → egy osztály lehetséges)
FOLDER_TO_CLASS = {
    "light_truck"       : "light_truck",
    "medium_truck"      : "heavy_truck",   # összevonva
    "heavy_truck"       : "heavy_truck",   # összevonva
    "vehicle_combination": "vehicle_combination",
    "bus_solo"          : "bus_solo",
    "bus_articulated"   : "bus_articulated",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ── Alapértelmezett útvonalak ──────────────────────────────────────────────────
DEFAULT_DATA_DIR   = r"C:\Users\admin\Trafic_mojo_2\1_classifier_data"
DEFAULT_OUTPUT_DIR = r"C:\Users\admin\Trafic_mojo_2\YOLO-training\models"

CROP_SIZE    = 224
MARGIN_FRAC  = 0.10   # 10%-os margó a bbox köré
TRAIN_RATIO  = 0.80   # 80% train, 20% val
SEED         = 42


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def crop_from_yolo(img: Image.Image, txt_path: Path) -> Image.Image | None:
    """
    Beolvassa a .txt YOLO annotációt (1 sor) és kivágja a bbox-ot a képből.
    Margót ad hozzá, majd CROP_SIZE×CROP_SIZE méretre méretezi.
    Visszatér None-nal, ha nincs .txt vagy üres.
    """
    if not txt_path.exists():
        return None
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None

    # Ha több sor van, a legnagyobb területű bbox-ot veszi (a fő jármű)
    best = None
    best_area = 0.0
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = map(float, parts)
        area = w * h
        if area > best_area:
            best_area = area
            best = (cx, cy, w, h)

    if best is None:
        return None

    cx, cy, w, h = best
    iw, ih = img.size

    # Margó hozzáadása
    w_m  = w  * (1 + 2 * MARGIN_FRAC)
    h_m  = h  * (1 + 2 * MARGIN_FRAC)

    x1 = max(0, int((cx - w_m / 2) * iw))
    y1 = max(0, int((cy - h_m / 2) * ih))
    x2 = min(iw, int((cx + w_m / 2) * iw))
    y2 = min(ih, int((cy + h_m / 2) * ih))

    if x2 <= x1 or y2 <= y1:
        return None

    return img.crop((x1, y1, x2, y2))


class VehicleDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        """
        samples: [(jpg_path, class_idx), ...]
        """
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jpg_path, label = self.samples[idx]
        txt_path = jpg_path.with_suffix(".txt")

        img = Image.open(jpg_path).convert("RGB")
        crop = crop_from_yolo(img, txt_path)
        if crop is None:
            # Fallback: egész kép
            crop = img

        return self.transform(crop), label


# ══════════════════════════════════════════════════════════════════════════════
# Adatkészítés
# ══════════════════════════════════════════════════════════════════════════════

def collect_samples(data_dir: Path) -> list[tuple[Path, int]]:
    samples = []
    for folder_name, cls_name in FOLDER_TO_CLASS.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            print(f"  [FIGYELEM] Hiányzó mappa: {folder_name}")
            continue
        # _rejected almappát kizárjuk
        jpgs = sorted([p for p in folder.glob("*.jpg")
                       if "_rejected" not in str(p)])
        for jpg in jpgs:
            samples.append((jpg, CLASS_TO_IDX[cls_name]))
    return samples


def split_samples(samples, train_ratio=TRAIN_RATIO, seed=SEED):
    rng = random.Random(seed)
    # Osztályonként stratifikált split
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train, val = [], []
    for idx, lst in by_class.items():
        rng.shuffle(lst)
        n_train = max(1, int(len(lst) * train_ratio))
        train.extend(lst[:n_train])
        val.extend(lst[n_train:])

    return train, val


def make_weighted_sampler(samples):
    labels   = [s[1] for s in samples]
    counts   = Counter(labels)
    n_total  = len(labels)
    weights  = [n_total / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ══════════════════════════════════════════════════════════════════════════════
# Modell
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# Tréning
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = Counter()
    per_class_total   = Counter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            per_class_total[l]   += 1
            per_class_correct[l] += int(p == l)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    return total_loss / total, correct / total, per_class_correct, per_class_total


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Vehicle sub-category classifier training")
    parser.add_argument("--data_dir",   default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch",      type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=4)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Vehicle Classifier – EfficientNet-B0")
    print(f"  Device  : {device}")
    print(f"  Classes : {CLASSES}")
    print(f"{'='*60}\n")

    # ── Adatok ────────────────────────────────────────────────────────────────
    all_samples = collect_samples(data_dir)
    counts = Counter(s[1] for s in all_samples)
    print("Képek osztályonként:")
    for cls_name, idx in CLASS_TO_IDX.items():
        # Megmutatja melyik mappákból jött
        src_folders = [f for f, c in FOLDER_TO_CLASS.items() if c == cls_name]
        src_str = "+".join(src_folders)
        print(f"  {cls_name:<22}: {counts[idx]:>4} kép  ({src_str})")
    print(f"  {'ÖSSZESEN':<22}: {len(all_samples):>4} kép\n")

    train_samples, val_samples = split_samples(all_samples)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}\n")

    # ── Transzformációk ────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((CROP_SIZE + 32, CROP_SIZE + 32)),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = VehicleDataset(train_samples, train_tf)
    val_ds   = VehicleDataset(val_samples,   val_tf)

    sampler    = make_weighted_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Modell ────────────────────────────────────────────────────────────────
    model     = build_model(len(CLASSES), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Tréning loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_path    = output_dir / "vehicle_classifier.pt"

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, pc_correct, pc_total = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        marker = " ★" if vl_acc > best_val_acc else ""
        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.1%} | {vl_loss:>8.4f} | {vl_acc:>8.1%} | {lr_now:>8.2e}{marker}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "model_state_dict" : model.state_dict(),
                "classes"          : CLASSES,
                "class_to_idx"     : CLASS_TO_IDX,
                "val_acc"          : vl_acc,
                "epoch"            : epoch,
                "crop_size"        : CROP_SIZE,
                "architecture"     : "efficientnet_b0",
            }, best_path)

    # ── Végeredmény ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Legjobb val accuracy: {best_val_acc:.1%}")
    print(f"  Mentve: {best_path}")
    print(f"{'='*60}\n")

    # Osztályonkénti pontosság a legjobb modellen
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, pc_correct, pc_total = evaluate(model, val_loader, criterion, device)
    print("Osztályonkénti Val pontosság:")
    for cls_name, idx in CLASS_TO_IDX.items():
        n = pc_total[idx]
        c = pc_correct[idx]
        acc = c / n if n > 0 else 0.0
        print(f"  {cls_name:<22}: {c:>3}/{n:<3}  ({acc:.0%})")


if __name__ == "__main__":
    main()
