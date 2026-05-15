#!/usr/bin/env python3
"""
Frame Extractor for YOLO Training Data
=======================================
Videókból kinyeri a kívánt kategóriájú objektumok legjobb frame-jeit
YOLO track() segítségével. Headless mód (nincs cv2.imshow), tqdm progress bar.

Működés:
  1. Betölti a meglévő traffic YOLO modellt
  2. Végigmegy a videó frame-jein, track()-kal nyomon követi az objektumokat
  3. Minden trigger-objektum ID-hoz eltárolja a legjobb pozíciót
     (konfidencia × képközeli súlyozással)
  4. Ha az ID eltűnik (v. videó végére ér): menti a képet + YOLO .txt-t
  5. Az összes detektált objektum belekerül a .txt-be (teljes annotáció!)
  6. Egy ID-ról max. --max_per_id kép készül → duplikáció szűrés

Trigger osztályok:
  --triggers  vesszővel elválasztott osztálynevek a modellből
              Ha nincs megadva → MINDEN osztály trigger lesz

Kimenet (egy mappa triggerenként):
  training_data/raw/
    person/
    car/
    bus/       ← review fogja bus_solo / bus_articulated-ra bontani
    truck/     ← review fogja truck_light / truck_heavy / vehicle_combination-ra bontani
    ...

Következő lépés:
  python scripts/review_annotations.py --source training_data/raw

Használat:
  # Minden osztály gyűjtése:
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt

  # Csak bus és truck:
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt --triggers bus,truck

  # Minden osztály, több kép/jármű, minden 3. frame:
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt --max_per_id 3 --frame_skip 3
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


# ── Osztályok ────────────────────────────────────────────────────────────────
# A modell saját names listáját használjuk (nincs hardcoded átindexelés).
# A kimeneti .txt osztály-ID-k megegyeznek a forrásmodell ID-jaival.
# A review_annotations.py fogja elvégezni a bus/truck alkategóriába sorolást.

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mts", ".m2ts", ".wmv", ".ts"}


# ── Trigger lista felépítése ─────────────────────────────────────────────────

def build_triggers(
    model_names: dict[int, str],
    requested: list[str] | None,
    quiet: bool = False,
) -> dict[int, str]:
    """
    Visszaad egy {model_class_id → folder_name} szótárt.

    Ha requested üres/None → minden modell-osztály trigger lesz.
    Ha requested meg van adva → csak azok, amik részstringjük megegyezik.

    A folder_name = modell osztályneve (kisbetűs, szóköz → _).
    Pl. model_names[5]='Bus' → folder='bus'
    """
    triggers: dict[int, str] = {}

    for cls_id, name in sorted(model_names.items()):
        norm = name.lower().replace(" ", "_")
        folder = norm

        if requested:
            matched = any(req in norm or norm in req for req in requested)
        else:
            matched = True

        if matched:
            triggers[cls_id] = folder

    if not triggers:
        print("[HIBA] Egyetlen trigger osztály sem maradt!")
        print(f"  Kért osztályok: {requested}")
        print(f"  Modell osztályok: {list(model_names.values())}")
        sys.exit(1)

    if not quiet:
        print(f"[OK] Aktív triggerek: {', '.join(triggers.values())}")
    return triggers


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
                "obj_id":      rec["obj_id"],
                "trigger_cls": rec["trigger_cls"],
                "folder":      rec["folder"],
                "best_score":  round(rec["best_score"], 4),
            },
            f,
            indent=2,
        )


# ── Videó feldolgozás ─────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    model: YOLO,
    triggers: dict[int, str],
    output_dir: Path,
    max_per_id: int,
    frame_skip: int,
    conf_thr: float,
    iou_thr: float,
    min_motion: float = 0.04,
) -> tuple[int, dict[str, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[HIBA] Nem nyitható: {video_path.name}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = video_path.stem.replace(" ", "_")

    # Tracker állapot resetelése az új videóhoz
    if hasattr(model, "predictor") and model.predictor is not None:
        try:
            model.predictor = None
        except Exception:
            pass

    tracked: dict[int, dict] = {}
    saved = 0
    skipped_stationary = 0
    saved_by_cls: dict[str, int] = {}
    fi = 0

    pbar = tqdm(
        total=total,
        desc=f"{video_path.name[:38]}",
        unit="f",
        dynamic_ncols=True,
        colour=None,
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

            # Összes jelen lévő box (modell eredeti ID-kkal – nincs átindexelés)
            all_boxes: list[tuple] = [
                (cls, cx, cy, bw, bh, cf)
                for cls, cf, (cx, cy, bw, bh) in zip(clss, confs, xywhn)
            ]

            # Trigger objektumok frissítése
            for oid, cls, cf, (cx, cy, bw, bh) in zip(ids, clss, confs, xywhn):
                if cls not in triggers:
                    continue

                current_ids.add(oid)

                dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
                score = cf * 0.7 + max(0.0, 1.0 - dist * 2.0) * 0.3

                if oid not in tracked:
                    folder = triggers[cls]
                    tracked[oid] = {
                        "obj_id":      oid,
                        "trigger_cls": cls,
                        "folder":      folder,
                        "best_score":  -1.0,
                        "best_frame":  None,
                        "best_boxes":  [],
                        "saved":       0,
                        "first_pos":   (cx, cy),   # első észlelt pozíció
                        "max_disp":    0.0,         # max elmozdulás az első pozíciótól
                    }

                rec = tracked[oid]
                if rec["saved"] >= max_per_id:
                    continue

                # Elmozdulás követése
                dx = cx - rec["first_pos"][0]
                dy = cy - rec["first_pos"][1]
                disp = math.sqrt(dx * dx + dy * dy)
                if disp > rec["max_disp"]:
                    rec["max_disp"] = disp

                if score > rec["best_score"]:
                    rec["best_score"] = score
                    rec["best_frame"] = frame.copy()
                    rec["best_boxes"] = list(all_boxes)

        # Eltűnt ID-k → mentés (csak ha eleget mozgott)
        gone = set(tracked.keys()) - current_ids
        for oid in list(gone):
            rec = tracked[oid]
            if rec["saved"] < max_per_id and rec["best_frame"] is not None:
                if rec["max_disp"] >= min_motion:
                    save_best_frame(rec, stem, fi, output_dir)
                    rec["saved"] += 1
                    saved += 1
                    saved_by_cls[rec["folder"]] = saved_by_cls.get(rec["folder"], 0) + 1
                    pbar.set_postfix(mentve=saved)
                else:
                    skipped_stationary += 1
            del tracked[oid]

    pbar.close()
    cap.release()

    # Videó vége: maradék tracker ID-k mentése
    for oid, rec in tracked.items():
        if rec["saved"] < max_per_id and rec["best_frame"] is not None:
            if rec["max_disp"] >= min_motion:
                save_best_frame(rec, stem, fi, output_dir)
                saved += 1
                saved_by_cls[rec["folder"]] = saved_by_cls.get(rec["folder"], 0) + 1
            else:
                skipped_stationary += 1

    print(f"  \u2192 {saved} kép mentve | {video_path.name} (mozgás alatt: {skipped_stationary} álló kihagyva)")
    cls_info = "  ".join(f"{k}: {v}" for k, v in sorted(saved_by_cls.items()))
    if cls_info:
        print(f"    {cls_info}")
    return saved, saved_by_cls


# ── Argumentumok ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frame Extractor – képek kinyerése videókból YOLO track()-kal (bármely kategória)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák:
  # Minden osztály gyűjtése (alapértelmezett):
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt

  # Csak bus és truck:
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt --triggers bus,truck

  # Minden osztály, 3 kép/jármű, minden 3. frame:
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt --max_per_id 3 --frame_skip 3

  # Csak person és car, CPU-n:
  python scripts/extract_frames.py --video D:/videos --model weights/best.pt --triggers person,car --device cpu
        """,
    )
    p.add_argument("--video", type=Path, required=True,
                   help="Videó fájl vagy videókat tartalmazó mappa (rekurzív)")
    p.add_argument("--model", type=Path, required=True,
                   help="Meglévő traffic YOLO modell súlyok (.pt)")
    p.add_argument("--triggers", type=str, default=None,
                   help="Vesszővel elválasztott trigger osztályok (pl. 'bus,truck,car'). "
                        "Ha nincs megadva → MINDEN osztály trigger lesz.")
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
    p.add_argument("--min_motion", type=float, default=0.04,
                   help="Minimális elmozdulás képmerethez viszonyítva (0–1, alapért.: 0.04). "
                        "Ez szűri ki a piroslámpán várakozó járműveket.")
    p.add_argument("--quiet", action="store_true",
                   help="Osztálylista és részletes modell-info elnyomása")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.model.exists():
        print(f"[HIBA] Modell nem található: {args.model}")
        sys.exit(1)

    if not args.quiet:
        print(f"[MODELL] Betöltés: {args.model}")
    model = YOLO(str(args.model))

    requested = (
        [t.strip().lower().replace(" ", "_") for t in args.triggers.split(",")]
        if args.triggers else None
    )
    triggers = build_triggers(model.names, requested, quiet=args.quiet)

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

    if not args.quiet:
        print(f"\n[OK] {len(videos)} videó feldolgozása → {args.output}\n")

    total_saved = 0
    total_by_cls: dict[str, int] = {}
    for v in videos:
        n, by_cls = process_video(
            v, model, triggers,
            args.output, args.max_per_id, args.frame_skip,
            args.conf, args.iou, args.min_motion,
        )
        total_saved += n
        for cls, cnt in by_cls.items():
            total_by_cls[cls] = total_by_cls.get(cls, 0) + cnt

    print(f"\n[KÉSZ] Összes mentett kép: {total_saved}")
    for cls in sorted(total_by_cls):
        print(f"       {cls:<22} {total_by_cls[cls]:>5} kép")
    print(f"       Kimenet: {args.output.resolve()}")
    print(f"       Következő: python scripts/review_annotations.py --source {args.output}")


if __name__ == "__main__":
    main()
