#!/usr/bin/env python3
"""
Dataset Cleaner and Merger
==========================
Kombinált COCO 2017 + egyedi adathalmaz előkészítő script.

Végleges osztálystruktúra:
  0: person              (COCO alap)
  1: bicycle             (COCO alap)
  2: car                 (COCO alap)
  3: motorcycle          (COCO alap)
  4: bus_solo            (Egyedi – szóló busz)
  5: bus_articulated     (Egyedi – csuklós busz)
  6: truck_light         (Egyedi – kisteher, furgon, < 3.5t)
  7: truck_heavy         (Egyedi – merev felépítményű teherautó, > 3.5t)
  8: vehicle_combination (Egyedi – nyerges vontatós, pótkocsival)

A COCO Roboflow exportban az osztályok NEM 0-3 indexen vannak!
A script automatikusan beolvassa a COCO data.yaml-t és elvégzi az átindexelést.

Használat:
  python scripts/prepare_dataset.py --coco_dir "D:/Traffic_Mojo_2/YOLO_training/Microsoft COCO dataset"
  python scripts/prepare_dataset.py --coco_dir "..." --custom_root "..." --output dataset
  python scripts/prepare_dataset.py --custom_root path/to/egyedi_kepek
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# ── Osztály-konfiguráció ──────────────────────────────────────────────────────

CLASS_NAMES = [
    "person",               # 0
    "bicycle",              # 1
    "car",                  # 2
    "motorcycle",           # 3
    "bus_solo",             # 4
    "bus_articulated",      # 5
    "truck_light",          # 6
    "truck_heavy",          # 7
    "vehicle_combination",  # 8
]

# Osztálynevek amit a COCO data.yaml-ban keresünk → célindex
COCO_NAME_TO_TARGET: dict[str, int] = {
    "person":     0,
    "bicycle":    1,
    "bike":       1,
    "car":        2,
    "automobile": 2,
    "motorcycle": 3,
    "motorbike":  3,
    "moped":      3,
}

# Mappanév → class ID az egyedi képekhez
CUSTOM_FOLDER_MAP: dict[str, int] = {
    # bus_solo (4)
    "bus_solo": 4, "solo_bus": 4, "szolo_busz": 4,
    "szolóbusz": 4, "solo_busz": 4,
    # bus_articulated (5)
    "bus_articulated": 5, "articulated_bus": 5, "csuklos_busz": 5,
    "csuklos_busz": 5, "csuklosbusz": 5,
    # truck_light (6)
    "truck_light": 6, "light_truck": 6, "konnyuteher": 6,
    "konnyu_teher": 6, "furgon": 6, "kisteher": 6,
    # truck_heavy (7)
    "truck_heavy": 7, "heavy_truck": 7, "nehezteher": 7,
    "nehezteher": 7, "kozepes_teher": 7, "kamion": 7,
    # vehicle_combination (8)
    "vehicle_combination": 8, "szerelveny": 8,
    "jarmuszerveny": 8, "trailer": 8, "nyerges": 8, "combination": 8,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# ── COCO ID-térkép felépítése data.yaml-ból ───────────────────────────────────

def build_coco_id_remap(coco_dir: Path) -> dict[int, int]:
    yaml_path = coco_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"[COCO] FIGYELEM: data.yaml nem találhato: {yaml_path}")
        print("[COCO] Standard COCO ID-k feltételezése: person=0,bicycle=1,car=2,motorbike=3")
        return {0: 0, 1: 1, 2: 2, 3: 3}

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    source_names: list[str] = data.get("names", [])
    remap: dict[int, int] = {}

    print(f"\n[COCO] data.yaml beolvasva: {yaml_path}")
    print(f"[COCO] Forrás osztályok száma: {len(source_names)}")

    for src_id, name in enumerate(source_names):
        normalized = name.lower().replace("-", "").replace(" ", "").replace("_", "")
        for pattern, tgt_id in COCO_NAME_TO_TARGET.items():
            if normalized == pattern.replace("-", "").replace("_", ""):
                remap[src_id] = tgt_id
                print(f"  [{src_id:>3}] {name:<20} -> [{tgt_id}] {CLASS_NAMES[tgt_id]}")
                break

    if not remap:
        print("[COCO] HIBA: Egyetlen releváns osztály sem találhato a data.yaml-ban!")
        sys.exit(1)

    return remap


# ── COCO labelek szűrése és átindexelése ─────────────────────────────────────

def filter_coco_labels(
    images_dir: Path,
    labels_dir: Path,
    id_remap: dict[int, int],
    max_per_class: int,
) -> list[dict]:
    print(f"\n[COCO] Képek:   {images_dir}")
    print(f"[COCO] Labelek: {labels_dir}")

    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        print("[COCO] FIGYELEM: Nem találhatók .txt fájlok a labels mappában!")
        return []

    samples: list[dict] = []
    skipped_empty = 0
    skipped_no_image = 0

    for label_path in label_files:
        kept_lines: list[str] = []
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    src_id = int(parts[0])
                    if src_id in id_remap:
                        tgt_id = id_remap[src_id]
                        kept_lines.append(f"{tgt_id} " + " ".join(parts[1:]))
        except (ValueError, IOError) as exc:
            print(f"[COCO] Hiba ({label_path.name}): {exc}")
            continue

        if not kept_lines:
            skipped_empty += 1
            continue

        img_path = None
        for ext in IMAGE_EXTS:
            candidate = images_dir / (label_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped_no_image += 1
            continue

        target_classes = {int(ln.split()[0]) for ln in kept_lines}
        samples.append({
            "image": img_path,
            "label": label_path,
            "classes": target_classes,
            "remapped_lines": kept_lines,
        })

    print(f"[COCO] Érvényes minták: {len(samples)}")
    print(f"[COCO] Kihagyva - üres: {skipped_empty}, hiányzó kép: {skipped_no_image}")

    return _balance_coco(samples, id_remap, max_per_class)


def _balance_coco(samples: list[dict], id_remap: dict[int, int], max_per_class: int) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    target_ids = set(id_remap.values())

    raw_counts: dict[int, int] = defaultdict(int)
    for s in samples:
        for cls_id in s["classes"]:
            raw_counts[cls_id] += 1

    print(f"\n[COCO] Képszámok (korlát: {max_per_class}/osztály):")
    for tgt_id in sorted(target_ids):
        print(f"  [{tgt_id}] {CLASS_NAMES[tgt_id]:<22} {raw_counts.get(tgt_id, 0):>5} kép")

    if all(raw_counts.get(t, 0) <= max_per_class for t in target_ids):
        print("[COCO] Balansz nem szükséges.")
        return samples

    rng.shuffle(samples)
    kept: list[dict] = []
    kept_counts: dict[int, int] = defaultdict(int)

    for s in samples:
        s_target_cls = s["classes"] & target_ids
        if all(kept_counts[c] < max_per_class for c in s_target_cls):
            kept.append(s)
            for c in s_target_cls:
                kept_counts[c] += 1

    print(f"[COCO] Balansz utáni minták: {len(kept)} (volt: {len(samples)})")
    return kept


# ── Egyedi képek feldolgozása ─────────────────────────────────────────────────

def process_custom_images(custom_root: Path) -> list[dict]:
    print(f"\n[EGYEDI] Gyökérmappa: {custom_root}")

    samples: list[dict] = []
    unknown_folders: list[str] = []

    for folder in sorted(custom_root.iterdir()):
        if not folder.is_dir():
            continue

        folder_key = folder.name.lower().replace(" ", "_")
        class_id = CUSTOM_FOLDER_MAP.get(folder_key)

        if class_id is None:
            unknown_folders.append(folder.name)
            continue

        images = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        print(f"  [{class_id}] {class_name:<22} <- {folder.name}: {len(images)} kép")

        for img_path in images:
            samples.append({
                "image": img_path,
                "label": None,
                "classes": {class_id},
                "remapped_lines": [f"{class_id} 0.5 0.5 1.0 1.0"],
            })

    if unknown_folders:
        print(f"\n[EGYEDI] FIGYELEM: Ismeretlen mappák (kihagyva): {unknown_folders}")
        print("  Add hozzá a CUSTOM_FOLDER_MAP szótárhoz a scriptben!")

    print(f"[EGYEDI] Feldolgozott egyedi minták: {len(samples)}")
    return samples


# ── Egyesítés, split és másolás ───────────────────────────────────────────────

def merge_and_split(all_samples: list[dict], output_dir: Path) -> None:
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_samples)

    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"\n[SPLIT] Összes: {len(all_samples)}  |  Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    for subset in ("train", "val"):
        (output_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    for subset, samples in [("train", train_samples), ("val", val_samples)]:
        img_out = output_dir / "images" / subset
        lbl_out = output_dir / "labels" / subset

        for s in samples:
            img_src: Path = s["image"]
            stem = f"{img_src.parent.name}_{img_src.stem}"
            shutil.copy2(img_src, img_out / (stem + img_src.suffix))
            lbl_dst = lbl_out / (stem + ".txt")
            with open(lbl_dst, "w", encoding="utf-8") as f:
                f.write("\n".join(s["remapped_lines"]) + "\n")

    print(f"[SPLIT] Fájlok kimásolva -> {output_dir}")


# ── data.yaml generálás ───────────────────────────────────────────────────────

def generate_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "data.yaml"
    data = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"\n[YAML] data.yaml generálva -> {yaml_path}")


# ── Statisztika ───────────────────────────────────────────────────────────────

def print_stats(samples: list[dict], label: str) -> None:
    counts: dict[int, int] = defaultdict(int)
    for s in samples:
        for cls_id in s["classes"]:
            counts[cls_id] += 1
    print(f"\n[STAT] {label}:")
    for cls_id in sorted(counts):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"  [{cls_id}] {name:<22} {counts[cls_id]:>5} kép")
    print(f"  {'ÖSSZESEN':<24} {sum(counts.values()):>5}")


# ── COCO split-ek felderítése ─────────────────────────────────────────────────

def find_coco_splits(coco_dir: Path, requested: str | None) -> list[tuple[Path, Path]]:
    KNOWN_SPLITS = ["train", "valid", "val", "test"]

    if requested:
        split_names = [s.strip() for s in requested.split(",")]
    else:
        split_names = []
        for s in KNOWN_SPLITS:
            if (coco_dir / s / "images").exists() or (coco_dir / "images" / s).exists():
                split_names.append(s)
        if not split_names:
            if (coco_dir / "images").exists() and (coco_dir / "labels").exists():
                return [(coco_dir / "images", coco_dir / "labels")]
        if not split_names:
            print(f"[COCO] FIGYELEM: Nem találhatók képmappák a {coco_dir} alatt")
            return []

    result: list[tuple[Path, Path]] = []
    for split in split_names:
        if (coco_dir / split / "images").exists():
            img_dir = coco_dir / split / "images"
            lbl_dir = coco_dir / split / "labels"
        elif (coco_dir / "images" / split).exists():
            img_dir = coco_dir / "images" / split
            lbl_dir = coco_dir / "labels" / split
        else:
            print(f"[COCO] FIGYELEM: '{split}' split nem található: {coco_dir}")
            continue

        if not lbl_dir.exists():
            print(f"[COCO] FIGYELEM: Labels mappa hiányzik: {lbl_dir}")
            continue

        print(f"[COCO] Split felismerve: {split}  ({img_dir})")
        result.append((img_dir, lbl_dir))

    return result


# ── Argumentumok ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset Cleaner and Merger - COCO + egyedi adatok összevonása"
    )
    parser.add_argument("--coco_dir", type=Path, default=None,
        help="COCO dataset gyökérmappája (tartalmazza a data.yaml-t és a split almappákat)")
    parser.add_argument("--coco_split", type=str, default=None,
        help="Melyik COCO split: 'train', 'valid', 'test', vagy 'train,valid'. Auto-detect ha nincs megadva.")
    parser.add_argument("--custom_root", type=Path, default=None,
        help="Egyedi képek gyökérmappája (almappák neve = osztálynév)")
    parser.add_argument("--output", type=Path, default=Path("dataset"),
        help="Kimeneti mappa (alapért.: ./dataset)")
    parser.add_argument("--max_coco_per_class", type=int, default=2000,
        help="Max COCO kép osztályonként balansz céljából (alapért.: 2000)")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    all_samples: list[dict] = []

    if args.coco_dir:
        if not args.coco_dir.exists():
            print(f"[HIBA] A mappa nem találhato (--coco_dir): {args.coco_dir}")
            sys.exit(1)
        id_remap = build_coco_id_remap(args.coco_dir)
        splits = find_coco_splits(args.coco_dir, args.coco_split)
        if not splits:
            print("[COCO] Nem sikerült COCO adatot betölteni.")
        else:
            for img_dir, lbl_dir in splits:
                coco_samples = filter_coco_labels(img_dir, lbl_dir, id_remap, args.max_coco_per_class)
                print_stats(coco_samples, f"COCO ({lbl_dir.parent.name})")
                all_samples.extend(coco_samples)
    else:
        print("[INFO] COCO adatok kihagyva (--coco_dir nem megadva)")

    if args.custom_root:
        if not args.custom_root.exists():
            print(f"[HIBA] A mappa nem találhato (--custom_root): {args.custom_root}")
            sys.exit(1)
        custom_samples = process_custom_images(args.custom_root)
        print_stats(custom_samples, "Egyedi képek")
        all_samples.extend(custom_samples)
    else:
        print("[INFO] Egyedi képek kihagyva (--custom_root nem megadva)")

    if not all_samples:
        print("\n[HIBA] Nincs feldolgozható adat.")
        print('Pelda: python scripts/prepare_dataset.py --coco_dir "D:/Traffic_Mojo_2/YOLO_training/Microsoft COCO dataset"')
        sys.exit(1)

    print_stats(all_samples, "TELJES adathalmaz (egyesített)")
    merge_and_split(all_samples, args.output)
    generate_yaml(args.output)

    print("\n[OK] Dataset előkészítés sikeresen kész!")
    print(f"     Kimenet: {args.output.resolve()}")
    print(f"     Tréning: python scripts/train.py")


if __name__ == "__main__":
    main()
