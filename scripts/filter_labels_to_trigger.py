"""
filter_labels_to_trigger.py
----------------------------
A Training_pictures_reviewed/ mappában minden almappához tartozik egy osztály.
Ez a script minden .txt annotációs fájlból csak a trigger osztályú sort(oka)t tartja meg.

Fontos: a review_annotations.py remap-elési bugja miatt előfordulhat, hogy
a trigger jármű rossz ID-val van elmentve (pl. heavy_truck mappában vehicle_combination
ID-val). Ezért a script "rokon osztály remap"-et alkalmaz:
- truck-féle mappákban (light/medium/heavy/vehicle_combination): az összes truck ID
  (4,5,6,7) átírásra kerül a mappanév szerinti helyes ID-ra
- bus-féle mappákban (bus_solo, bus_articulated, minibus): bus ID-k (8,9,13) átírva
- A többi mappa esetén: csak a pontos ID-egyezés marad meg

Eredmény: minden képhez csak az ellenőrzött trigger jármű annotációja marad,
helyes osztály ID-val.

Használat:
  python scripts/filter_labels_to_trigger.py
  python scripts/filter_labels_to_trigger.py --reviewed_root "C:/Users/admin/.../Training_pictures_reviewed" --dry_run
"""

import argparse
import pathlib

# Osztálynév → ID leképezés (data.yaml szerint)
CLASS_NAME_TO_ID = {
    "person":               0,
    "bicycle":              1,
    "motorcycle":           2,
    "personal_car":         3,
    "light_truck":          4,
    "medium_truck":         5,
    "heavy_truck":          6,
    "vehicle_combination":  7,
    "bus_solo":             8,
    "bus_articulated":      9,
    "trolley_solo":         10,
    "trolley_articulated":  11,
    "tram":                 12,
    "minibus":              13,
}

# Rokon osztályok: ha a mappanév truck-féle, az összes truck ID átírható
# a felhasználó döntésének megfelelő ID-ra
REMAP_GROUPS = {
    # truck-féle mappák: ha bármely truck ID szerepel, átírjuk a helyes ID-ra
    "light_truck":          {4, 5, 6, 7},
    "medium_truck":         {4, 5, 6, 7},
    "heavy_truck":          {4, 5, 6, 7},
    "vehicle_combination":  {4, 5, 6, 7},
    # bus-féle mappák
    "bus_solo":             {8, 9, 13},
    "bus_articulated":      {8, 9, 13},
    "minibus":              {8, 9, 13},
}

# Ezeket a mappákat hagyjuk ki
SKIP_FOLDERS = {"skipped", "dataset", "deleted"}


def filter_folder(folder: pathlib.Path, class_id: int, dry_run: bool) -> dict:
    stats = {"kept": 0, "removed_lines": 0, "remapped_lines": 0,
             "empty_files": 0, "errors": 0, "txt_files": 0}

    folder_name = folder.name
    remap_group = REMAP_GROUPS.get(folder_name)  # None ha nincs rokon csoport

    txt_files = list(folder.glob("*.txt"))
    stats["txt_files"] = len(txt_files)

    for txt_path in txt_files:
        try:
            lines = txt_path.read_text(encoding="utf-8").splitlines()
            kept_lines = []
            removed = 0
            remapped = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    file_class_id = int(parts[0])
                except ValueError:
                    continue

                if file_class_id == class_id:
                    # Pontosan egyező ID – megtartjuk
                    kept_lines.append(line)
                elif remap_group and file_class_id in remap_group:
                    # Rokon ID – átírjuk a helyes ID-ra (felhasználó döntése alapján)
                    parts[0] = str(class_id)
                    kept_lines.append(" ".join(parts))
                    remapped += 1
                else:
                    removed += 1

            stats["removed_lines"] += removed
            stats["remapped_lines"] += remapped

            if kept_lines:
                stats["kept"] += 1
                if not dry_run:
                    txt_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
            else:
                stats["empty_files"] += 1
                if not dry_run:
                    txt_path.write_text("", encoding="utf-8")

        except Exception as e:
            stats["errors"] += 1
            print(f"  HIBA: {txt_path.name}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Annotációs szűrő – csak trigger osztály megtartása")
    parser.add_argument(
        "--reviewed_root",
        default=r"C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed",
        help="A reviewed mappa elérési útja"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Csak statisztikát mutat, nem módosít fájlokat"
    )
    args = parser.parse_args()

    root = pathlib.Path(args.reviewed_root)
    if not root.exists():
        print(f"HIBA: Nem található: {root}")
        return

    if args.dry_run:
        print("=== DRY RUN – nem módosít fájlokat ===\n")
    else:
        print("=== ÉLES FUTÁS – fájlok módosítása! ===\n")

    total_removed = 0
    total_kept = 0
    total_empty = 0

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        folder_name = folder.name

        if folder_name in SKIP_FOLDERS:
            print(f"[KIHAGYVA] {folder_name}/")
            continue

        if folder_name not in CLASS_NAME_TO_ID:
            print(f"[ISMERETLEN] {folder_name}/ – nincs osztály ID hozzá, kihagyva")
            continue

        class_id = CLASS_NAME_TO_ID[folder_name]
        stats = filter_folder(folder, class_id, args.dry_run)

        status = "DRY" if args.dry_run else "OK"
        remap_note = f" | átírt sor: {stats['remapped_lines']}" if stats['remapped_lines'] else ""
        print(
            f"[{status}] {folder_name}/ (ID={class_id}) | "
            f"{stats['txt_files']} fájl | "
            f"megtartott: {stats['kept']} | "
            f"törölt sor: {stats['removed_lines']}"
            + remap_note +
            f" | üres lett: {stats['empty_files']}"
            + (f" | HIBA: {stats['errors']}" if stats['errors'] else "")
        )

        total_removed += stats["removed_lines"]
        total_kept += stats["kept"]
        total_empty += stats["empty_files"]
        total_remapped = locals().get("total_remapped", 0) + stats["remapped_lines"]

    print(f"\n{'='*60}")
    print(f"Összesítés:")
    print(f"  Megtartott fájl:        {total_kept}")
    print(f"  Törölt annotáció sor:   {total_removed}")
    print(f"  Átírt ID (rokon→helyes):{total_remapped}  ← review_annotations.py bug javítva")
    print(f"  Üres fájl lett:         {total_empty}  (sem pontos, sem rokon ID nem volt)")
    if total_empty > 0:
        print(f"  → Az üres fájlokhoz tartozó képeket a prepare_dataset.py kiszűri.")
    if args.dry_run:
        print("\nFuttasd --dry_run nélkül a tényleges módosításhoz.")


if __name__ == "__main__":
    main()
