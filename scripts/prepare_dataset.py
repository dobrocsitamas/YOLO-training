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

Működés:
  1. COCO labelek szűrése: csak 0,1,2,3 ID-k maradnak; üres label + kép törlése.
  2. Egyedi képek feldolgozása: mappanév → class ID; teljes képet lefedő bbox.
  3. COCO minták balanszolása: max N kép osztályonként.
  4. 80/20 train/val split.
  5. Fájlok kimásolása a kimeneti struktúrába.
  6. data.yaml generálása.

Használat:
  python scripts/prepare_dataset.py \\
      --coco_images  path/to/coco/images \\
      --coco_labels  path/to/coco/labels \\
      --custom_root  path/to/custom_classes \\
      --output       dataset \\
      --max_coco_per_class 2000

  # Csak egyedi képek (COCO nélkül):
  python scripts/prepare_dataset.py --custom_root path/to/custom_classes
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
    "person",               # 0 – COCO alap
    "bicycle",              # 1 – COCO alap
    "car",                  # 2 – COCO alap
    "motorcycle",           # 3 – COCO alap
    "bus_solo",             # 4 – Egyedi
    "bus_articulated",      # 5 – Egyedi (csuklós busz)
    "truck_light",          # 6 – Egyedi (kisteher, furgon)
    "truck_heavy",          # 7 – Egyedi (merev felépítményű)
    "vehicle_combination",  # 8 – Egyedi (szerelvény, nyerges)
]

COCO_KEEP_IDS = frozenset({0, 1, 2, 3})

# Mappanév → class ID az egyedi képekhez (kisbetűs, szóköz → _)
CUSTOM_FOLDER_MAP: dict[str, int] = {
    # bus_solo (4)
    "bus_solo": 4,
    "solo_bus": 4,
    "szolo_busz": 4,
    "szólóbusz": 4,
    # bus_articulated (5)
    "bus_articulated": 5,
    "articulated_bus": 5,
    "csuklos_busz": 5,
    "csuklós_busz": 5,
    "csuklosbusz": 5,
    # truck_light (6)
    "truck_light": 6,
    "light_truck": 6,
    "konnyuteher": 6,
    "könnyűteher": 6,
    "könnyü_teher": 6,
    "furgon": 6,
    "kisteher": 6,
    # truck_heavy (7)
    "truck_heavy": 7,
    "heavy_truck": 7,
    "nehezteher": 7,
    "nehéz_teher": 7,
    "nehézteher": 7,
    "kozepes_teher": 7,
    "közepes_teher": 7,
    # vehicle_combination (8)
    "vehicle_combination": 8,
    "szerelveny": 8,
    "szerelvény": 8,
    "jarmuszerveny": 8,
    "trailer": 8,
    "nyerges": 8,
    "nyerges_vontatós": 8,
    "combination": 8,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# ── COCO szűrés ───────────────────────────────────────────────────────────────

def filter_coco_labels(
    images_dir: Path,
    labels_dir: Path,
    max_per_class: int,
) -> list[dict]:
    """
    Szűri a COCO labeleket: csak COCO_KEEP_IDS (0–3) sorok maradnak.
    Törli az üres label fájlokat és a hozzájuk tartozó képeket.
    Visszaadja az érvényes minták listáját.
    """
    print(f"\n[COCO] Képek:   {images_dir}")
    print(f"[COCO] Labelek: {labels_dir}")

    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        print("[COCO] FIGYELEM: Nem találhatók .txt fájlok a labels mappában!")
        return []

    samples: list[dict] = []
    deleted_empty = 0
    deleted_no_image = 0

    for label_path in label_files:
        # --- Szűrés: csak COCO_KEEP_IDS sorok ---
        kept_lines: list[str] = []
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts and int(parts[0]) in COCO_KEEP_IDS:
                        kept_lines.append(line)
        except (ValueError, IOError) as exc:
            print(f"[COCO] Hiba ({label_path.name}): {exc}")
            continue

        # Üres label → törlés
        if not kept_lines:
            label_path.unlink(missing_ok=True)
            for ext in IMAGE_EXTS:
                img = images_dir / (label_path.stem + ext)
                if img.exists():
                    img.unlink()
                    break
            deleted_empty += 1
            continue

        # Label fájl visszaírása a szűrt sorokkal
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_lines) + "\n")

        # Képfájl keresése
        img_path: Path | None = None
        for ext in IMAGE_EXTS:
            candidate = images_dir / (label_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            label_path.unlink(missing_ok=True)
            deleted_no_image += 1
            continue

        classes_in_image = {int(ln.split()[0]) for ln in kept_lines}
        samples.append({"image": img_path, "label": label_path, "classes": classes_in_image})

    print(f"[COCO] Érvényes minták: {len(samples)}")
    print(f"[COCO] Törölt – üres label: {deleted_empty}, hiányzó kép: {deleted_no_image}")

    return _balance_coco(samples, max_per_class)


def _balance_coco(samples: list[dict], max_per_class: int) -> list[dict]:
    """Véletlenszerűen csökkenti a COCO mintákat max_per_class darabra osztályonként."""
    rng = random.Random(RANDOM_SEED)

    raw_counts: dict[int, int] = defaultdict(int)
    for s in samples:
        for cls_id in s["classes"]:
            if cls_id in COCO_KEEP_IDS:
                raw_counts[cls_id] += 1

    print(f"[COCO] Képszámok osztályonként (korlát: {max_per_class}/osztály):")
    for cls_id in sorted(COCO_KEEP_IDS):
        print(f"  {cls_id}: {CLASS_NAMES[cls_id]:<14} {raw_counts[cls_id]:>5} kép")

    if all(v <= max_per_class for v in raw_counts.values()):
        print("[COCO] Balansz nem szükséges.")
        return samples

    rng.shuffle(samples)
    kept: list[dict] = []
    kept_counts: dict[int, int] = defaultdict(int)

    for s in samples:
        coco_cls = s["classes"] & COCO_KEEP_IDS
        if all(kept_counts[c] < max_per_class for c in coco_cls):
            kept.append(s)
            for c in coco_cls:
                kept_counts[c] += 1

    print(f"[COCO] Balansz utáni minták: {len(kept)} (volt: {len(samples)})")
    return kept


# ── Egyedi képek feldolgozása ─────────────────────────────────────────────────

def process_custom_images(custom_root: Path) -> list[dict]:
    """
    Egyedi képek feldolgozása mappanév → class ID alapján.
    Minden képhez létrehoz egy YOLO .txt fájlt (teljes képet lefedő bbox:
    class_id 0.5 0.5 1.0 1.0).
    """
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
        print(f"  [{class_id}: {class_name:<22}] {folder.name}: {len(images)} kép")

        for img_path in images:
            label_path = img_path.with_suffix(".txt")
            # YOLO formátum: class_id cx cy w h  (teljes kép)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            samples.append({
                "image": img_path,
                "label": label_path,
                "classes": {class_id},
            })

    if unknown_folders:
        print(f"\n[EGYEDI] FIGYELEM: Ismeretlen mappák (kihagyva): {unknown_folders}")
        print("  Adj hozzá bejegyzést a CUSTOM_FOLDER_MAP szótárhoz a scriptben!")

    print(f"[EGYEDI] Feldolgozott egyedi minták: {len(samples)}")
    return samples


# ── Egyesítés, split és másolás ───────────────────────────────────────────────

def merge_and_split(all_samples: list[dict], output_dir: Path) -> None:
    """80/20 train/val split, majd kimásolás a végleges struktúrába."""
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_samples)

    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"\n[SPLIT] Összes: {len(all_samples)}  |  Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    # Könyvtárstruktúra
    for subset in ("train", "val"):
        (output_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    for subset, samples in [("train", train_samples), ("val", val_samples)]:
        img_out = output_dir / "images" / subset
        lbl_out = output_dir / "labels" / subset

        for s in samples:
            img_src: Path = s["image"]
            lbl_src: Path = s["label"]
            # Prefix a szülőmappa nevéből → névütközés elkerülése
            stem = f"{img_src.parent.name}_{img_src.stem}"
            shutil.copy2(img_src, img_out / (stem + img_src.suffix))
            shutil.copy2(lbl_src, lbl_out / (stem + ".txt"))

    print(f"[SPLIT] Fájlok kimásolva → {output_dir}")


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
    print(f"\n[YAML] data.yaml generálva → {yaml_path}")


# ── Statisztika ───────────────────────────────────────────────────────────────

def _print_stats(samples: list[dict], label: str) -> None:
    counts: dict[int, int] = defaultdict(int)
    for s in samples:
        for cls_id in s["classes"]:
            counts[cls_id] += 1
    print(f"\n[STAT] {label}:")
    for cls_id in sorted(counts):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"  {cls_id}: {name:<22} {counts[cls_id]:>5} kép")
    print(f"  {'ÖSSZESEN':<24} {sum(counts.values()):>5}")


# ── Argumentumok ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset Cleaner and Merger – COCO + egyedi adatok összevonása"
    )
    parser.add_argument(
        "--coco_images", type=Path, default=None,
        help="COCO képek mappája (pl. coco2017/images/train2017)",
    )
    parser.add_argument(
        "--coco_labels", type=Path, default=None,
        help="COCO labelek mappája (pl. coco2017/labels/train2017)",
    )
    parser.add_argument(
        "--custom_root", type=Path, default=None,
        help="Egyedi képek gyökérmappája (almappák neve = osztálynév)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset"),
        help="Kimeneti mappa (alapért.: ./dataset)",
    )
    parser.add_argument(
        "--max_coco_per_class", type=int, default=2000,
        help="Max COCO kép osztályonként (balansz, alapért.: 2000)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    all_samples: list[dict] = []

    # --- COCO ---
    if args.coco_images and args.coco_labels:
        for p, name in [(args.coco_images, "coco_images"), (args.coco_labels, "coco_labels")]:
            if not p.exists():
                print(f"[HIBA] A mappa nem található (--{name}): {p}")
                sys.exit(1)
        coco_samples = filter_coco_labels(
            args.coco_images, args.coco_labels, args.max_coco_per_class
        )
        _print_stats(coco_samples, "COCO (szűrt + balanszolt)")
        all_samples.extend(coco_samples)
    else:
        print("[INFO] COCO adatok kihagyva (--coco_images / --coco_labels nem megadva)")

    # --- Egyedi ---
    if args.custom_root:
        if not args.custom_root.exists():
            print(f"[HIBA] A mappa nem található (--custom_root): {args.custom_root}")
            sys.exit(1)
        custom_samples = process_custom_images(args.custom_root)
        _print_stats(custom_samples, "Egyedi képek")
        all_samples.extend(custom_samples)
    else:
        print("[INFO] Egyedi képek kihagyva (--custom_root nem megadva)")

    if not all_samples:
        print("[HIBA] Nincs feldolgozható adat. Ellenőrizd a paramétereket.")
        sys.exit(1)

    _print_stats(all_samples, "TELJES adathalmaz (egyesített)")
    merge_and_split(all_samples, args.output)
    generate_yaml(args.output)

    print("\n[OK] Dataset előkészítés sikeresen kész!")
    print(f"     Tréning indítása: python scripts/train.py")


if __name__ == "__main__":
    main()
