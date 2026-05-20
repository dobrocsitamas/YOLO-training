"""
check_and_delete_empty.py
--------------------------
Megkeresi az üres .txt annotációs fájlokat a reviewed mappában,
és törli őket + a hozzájuk tartozó képfájlokat.

Használat:
  python scripts/check_and_delete_empty.py --dry_run   (csak listázás)
  python scripts/check_and_delete_empty.py             (törlés)
"""

import argparse
import pathlib

root = pathlib.Path(r"C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed")

CLASS_NAME_TO_ID = {
    "person":0,"bicycle":1,"motorcycle":2,"personal_car":3,
    "light_truck":4,"medium_truck":5,"heavy_truck":6,"vehicle_combination":7,
    "bus_solo":8,"bus_articulated":9,"trolley_solo":10,"trolley_articulated":11,
    "tram":12,"minibus":13,
}
SKIP = {"skipped","dataset","deleted"}

parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

if args.dry_run:
    print("=== DRY RUN – nem töröl ===\n")
else:
    print("=== TÖRLÉS ===\n")

total_deleted = 0
total_id_errors = 0

for folder in sorted(root.iterdir()):
    if not folder.is_dir() or folder.name in SKIP:
        continue
    class_id = CLASS_NAME_TO_ID.get(folder.name)
    if class_id is None:
        continue

    folder_deleted = 0
    id_errors = 0

    for txt in sorted(folder.glob("*.txt")):
        content = txt.read_text(encoding="utf-8").strip()
        if not content:
            # Üres – töröljük a txt-t és a képet
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = txt.with_suffix(ext)
                if candidate.exists():
                    img = candidate
                    break
            if args.dry_run:
                print(f"  TÖRLENDŐ: {txt.name}" + (f" + {img.name}" if img else " (kép nem található)"))
            else:
                txt.unlink()
                if img:
                    img.unlink()
            folder_deleted += 1
        else:
            # ID ellenőrzés
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        if int(parts[0]) != class_id:
                            id_errors += 1
                            print(f"  IDEGEN ID {parts[0]} (vart {class_id}): {txt.name}")
                    except ValueError:
                        pass

    status = "OK" if folder_deleted == 0 and id_errors == 0 else ("DRY" if args.dry_run else "TOROLT")
    print(f"[{status}] {folder.name}/ | törölve: {folder_deleted} | idegen ID: {id_errors}")
    total_deleted += folder_deleted
    total_id_errors += id_errors

print(f"\nÖsszesítés: {'törölve' if not args.dry_run else 'törlendő'}: {total_deleted} pár | idegen ID maradt: {total_id_errors}")
