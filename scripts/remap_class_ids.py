"""
remap_class_ids.py
------------------
A trolley_solo (10) és trolley_articulated (11) osztályok eltávolítása után
a tram (12→10) és minibus (13→11) ID-kat átírja.

Érinti:
  - dataset/labels/train/ és val/
  - Training_pictures_reviewed/tram/ és minibus/
"""
import pathlib

REMAP = {12: 10, 13: 11}

FOLDERS = [
    r"C:\Users\admin\Trafic_mojo_2\YOLO-training\dataset\labels\train",
    r"C:\Users\admin\Trafic_mojo_2\YOLO-training\dataset\labels\val",
    r"C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed\tram",
    r"C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed\minibus",
]

total_files = 0
total_lines = 0

for folder_str in FOLDERS:
    folder = pathlib.Path(folder_str)
    if not folder.exists():
        print(f"[KIHAGYVA – nem létezik] {folder}")
        continue
    changed_files = 0
    changed_lines = 0
    for txt in folder.glob("*.txt"):
        lines = txt.read_text(encoding="utf-8").splitlines()
        new_lines = []
        file_changed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            if cid in REMAP:
                parts[0] = str(REMAP[cid])
                changed_lines += 1
                file_changed = True
            new_lines.append(" ".join(parts))
        if file_changed:
            txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            changed_files += 1
    print(f"[OK] {folder.name}/ | módosított fájl: {changed_files} | átírt sor: {changed_lines}")
    total_files += changed_files
    total_lines += changed_lines

print(f"\nÖsszesen: {total_files} fájl, {total_lines} sor átírva (12→10, 13→11)")
