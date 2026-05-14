#!/usr/bin/env python3
"""
Frame Extractor for YOLO Training Data
=======================================
Videókból kinyeri a bus/truck detektálásokat YOLO track() segítségével.
Headless mód (nincs cv2.imshow), tqdm progress bar a konzolon.

Működés:
  1. Betölti a meglévő traffic YOLO modellt
  2. Végigmegy a videó frame-jein, tracked() segítségével nyomon követi az obj.
  3. Minden bus/truck ID-hoz eltárolja a legjobb pozíciót (konfidencia × képközeli)
  4. Ha az ID eltűnik (v. videó végére ér): menti a képet + YOLO .txt-t
  5. Az összes detektált objektum belekerül a .txt-be (teljes annotáció)
  6. Egy ID-ról max. 1 (--max_per_id) kép készül → duplikáció szűrés

Kimenet:
  training_data/raw/
    bus/    ← bus kivált képek + .txt + .json (metaadat)
    truck/  ← truck kivált képek + .txt + .json

Következő lépés:
  python scripts/review_annotations.py --source training_data/raw

Használat:
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt
  python scripts/extract_frames.py --video D:/videos/traffic.mp4 --model weights/best.pt --max_per_id 2 --frame_skip 3
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


# ── Osztályok (célmodell, 0–8) ────────────────────────────────────────────────

CLASS_NAMES_NEW = [
    "person",               # 0
    "bicycle",              # 1
    "car",                  # 2
    "motorcycle",           # 3
    "bus_solo",             # 4  ← placeholder (bus → review javítja)
    "bus_articulated",      # 5
    "truck_light",          # 6
    "truck_heavy",          # 7  ← placeholder (truck → review javítja)
    "vehicle_combination",  # 8
]

# Régi modell osztálynév (normalizált) → új class ID
OLD_NAME_TO_NEW: dict[str, int] = {
    "person":     0,
    "bicycle":    1,
    "bike":       1,
    "car":        2,
    "motorcycle": 3,
    "motorbike":  3,
    "bus":        4,   # placeholder → bus_solo, review fogja javítani
    "truck":      7,   # placeholder → truck_heavy, review fogja javítani
}

# Trigger osztályok: ha ilyen van a frame-ben → mentés
# kulcs: régi modell osztálynév (normalizált részstring)
# érték: (kimeneti mappa neve, placeholder new class ID)
TRIGGER_PATTERNS: dict[str, tuple[str, int]] = {
    "bus":   ("bus",   4),
    "truck": ("truck", 7),
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mts", ".m2ts", ".wmv", ".ts"}


# ── Osztálytérkép automatikus felépítése ──────────────────────────────────────

def build_remap(
    model_names: dict[int, str],
) -> tuple[dict[int, int], dict[int, tuple[str, int]]]:
    """
    model.names alapján felépíti:
      remap:    {old_id → new_id}        – minden felismert osztályra
      triggers: {old_id → (folder, placeholder_new_id)} – csak trigger osztályokra
    """
    remap: dict[int, int] = {}
    triggers: dict[int, tuple[str, int]] = {}

    print("\n[MODELL] Osztályok felismerése:")
    for old_id, name in sorted(model_names.items()):
        norm = name.lower().strip()
        matched = False

        for pattern, (folder, pholder) in TRIGGER_PATTERNS.items():
            if pattern in norm:
                remap[old_id] = pholder
                triggers[old_id] = (folder, pholder)
                print(f"  [{old_id:>3}] {name:<18} → TRIGGER → '{folder}/'  (placeholder={pholder})")
                matched = True
                break

        if not matched:
            new_id = OLD_NAME_TO_NEW.get(norm)
            if new_id is not None:
                remap[old_id] = new_id
                print(f"  [{old_id:>3}] {name:<18} → [{new_id}] {CLASS_NAMES_NEW[new_id]}")

    if not triggers:
        print("\n[HIBA] Nem található trigger osztály (bus/truck) a modellben!")
        print(f"  Modell osztályok: {list(model_names.values())}")
        sys.exit(1)

    unmapped = [name for old_id, name in model_names.items() if old_id not in remap]
    if unmapped:
        print(f"  [FIGYELEM] Nem leképezett osztályok (kihagyva): {unmapped}")

    return remap, triggers


# ── Kép + annotáció mentése ───────────────────────────────────────────────────

def save_best_frame(
    rec: dict,
    video_stem: str,
    frame_idx: int,
    output_dir: Path,
) -> None:
    """Elmenti a legjobb frame-et képként, YOLO .txt-ként és .json metaadat-ként."""
    out = output_dir / rec["folder"]
    out.mkdir(parents=True, exist_ok=True)

    stem = f"{video_stem}_{frame_idx:08d}_id{rec['obj_id']}"

    # Kép
    cv2.imwrite(
        str(out / f"{stem}.jpg"),
        rec["best_frame"],
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # YOLO .txt – az összes detektált objektum, átindexelve
    with open(out / f"{stem}.txt", "w", encoding="utf-8") as f:
        for (cls, cx, cy, w, h, _conf) in rec["best_boxes"]:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # Metaadat – a review_annotations.py fogja olvasni
    with open(out / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "obj_id":            rec["obj_id"],
                "trigger_old_cls":   rec["old_cls"],
                "placeholder_new_id": rec["pholder"],
                "folder":            rec["folder"],
                "best_score":        round(rec["best_score"], 4),
            },
            f,
            indent=2,
        )


# ── Videó feldolgozás ─────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    model: YOLO,
    remap: dict[int, int],
    triggers: dict[int, tuple[str, int]],
    output_dir: Path,
    max_per_id: int,
    frame_skip: int,
    conf_thr: float,
    iou_thr: float,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[HIBA] Nem nyitható: {video_path.name}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = video_path.stem.replace(" ", "_")

    # Tracker állapot resetelése az új videóhoz
    if hasattr(model, "predictor") and model.predictor is not None:
        try:
            model.predictor.trackers = []
        except Exception:
            pass

    tracked: dict[int, dict] = {}
    saved = 0
    fi = 0

    pbar = tqdm(
        total=total,
        desc=f"{video_path.name[:38]}",
        unit="f",
        dynamic_ncols=True,
        colour="cyan",
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        pbar.update(1)

        if fi % frame_skip != 0:
            continue

        results = model.track(
            source=frame,
            persist=True,
            conf=conf_thr,
            iou=iou_thr,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        boxes = results[0].boxes
        current_ids: set[int] = set()

        if boxes is not None and boxes.id is not None:
            ids   = boxes.id.int().cpu().tolist()
            clss  = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.float().cpu().tolist()
            xywhn = boxes.xywhn.float().cpu().tolist()  # [cx, cy, w, h] normalizálva

            # Összes jelen lévő box átindexelve
            all_boxes: list[tuple] = []
            for oid, cls, cf, (cx, cy, bw, bh) in zip(ids, clss, confs, xywhn):
                new_cls = remap.get(cls)
                if new_cls is not None:
                    all_boxes.append((new_cls, cx, cy, bw, bh, cf))

            # Trigger objektumok frissítése
            for oid, cls, cf, (cx, cy, bw, bh) in zip(ids, clss, confs, xywhn):
                if cls not in triggers:
                    continue

                current_ids.add(oid)

                dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
                score = cf * 0.7 + max(0.0, 1.0 - dist * 2.0) * 0.3

                if oid not in tracked:
                    folder, pholder = triggers[cls]
                    tracked[oid] = {
                        "obj_id":     oid,
                        "old_cls":    cls,
                        "folder":     folder,
                        "pholder":    pholder,
                        "best_score": -1.0,
                        "best_frame": None,
                        "best_boxes": [],
                        "saved":      0,
                    }

                rec = tracked[oid]
                if rec["saved"] >= max_per_id:
                    continue

                if score > rec["best_score"]:
                    rec["best_score"] = score
                    rec["best_frame"] = frame.copy()
                    rec["best_boxes"] = list(all_boxes)

        # Eltűnt ID-k → mentés
        gone = set(tracked.keys()) - current_ids
        for oid in list(gone):
            rec = tracked[oid]
            if rec["saved"] < max_per_id and rec["best_frame"] is not None:
                save_best_frame(rec, stem, fi, output_dir)
                rec["saved"] += 1
                saved += 1
                pbar.set_postfix(mentve=saved)
            del tracked[oid]

    pbar.close()
    cap.release()

    # Videó vége: maradék tracker ID-k mentése
    for oid, rec in tracked.items():
        if rec["saved"] < max_per_id and rec["best_frame"] is not None:
            save_best_frame(rec, stem, fi, output_dir)
            saved += 1

    print(f"  → {saved} kép mentve | {video_path.name}")
    return saved


# ── Argumentumok ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frame Extractor – bus/truck képek kinyerése videókból YOLO track()-kal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák:
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt
  python scripts/extract_frames.py --video D:/videos/cam1.mp4 --model weights/best.pt --max_per_id 2
        """,
    )
    p.add_argument("--video", type=Path, required=True,
                   help="Videó fájl vagy videókat tartalmazó mappa (rekurzív)")
    p.add_argument("--model", type=Path, required=True,
                   help="Meglévő traffic YOLO modell súlyok (.pt)")
    p.add_argument("--output", type=Path, default=Path("training_data/raw"),
                   help="Kimeneti mappa (alapért.: training_data/raw)")
    p.add_argument("--max_per_id", type=int, default=1,
                   help="Max kép egy tracked ID-ról (alapért.: 1)")
    p.add_argument("--frame_skip", type=int, default=2,
                   help="Minden N. frame-en fut a detektor (alapért.: 2)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Konfidencia küszöb (alapért.: 0.35)")
    p.add_argument("--iou", type=float, default=0.50,
                   help="IoU küszöb (alapért.: 0.50)")
    p.add_argument("--device", type=str, default="0",
                   help="Feldolgozó eszköz: '0' = GPU, 'cpu' (alapért.: 0)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.model.exists():
        print(f"[HIBA] Modell nem található: {args.model}")
        sys.exit(1)

    print(f"[MODELL] Betöltés: {args.model}")
    model = YOLO(str(args.model))

    remap, triggers = build_remap(model.names)

    # Videók összegyűjtése
    if args.video.is_dir():
        videos = sorted(
            f for f in args.video.rglob("*") if f.suffix.lower() in VIDEO_EXTS
        )
    elif args.video.exists():
        videos = [args.video]
    else:
        print(f"[HIBA] Nem található: {args.video}")
        sys.exit(1)

    if not videos:
        print(f"[HIBA] Nem találhatók videók ({', '.join(VIDEO_EXTS)}): {args.video}")
        sys.exit(1)

    print(f"\n[OK] {len(videos)} videó feldolgozása → {args.output}\n")

    total_saved = 0
    for v in videos:
        total_saved += process_video(
            v, model, remap, triggers,
            args.output, args.max_per_id, args.frame_skip,
            args.conf, args.iou,
        )

    print(f"\n[KÉSZ] Összes mentett kép: {total_saved}")
    print(f"       Kimenet: {args.output.resolve()}")
    print(f"       Következő: python scripts/review_annotations.py --source {args.output}")


if __name__ == "__main__":
    main()
