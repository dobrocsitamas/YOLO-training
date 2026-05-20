import glob, os
from collections import Counter
import openpyxl

folder = r"C:\Users\admin\Trafic_mojo_2\Eredmények\Teszt_YOLO11S"
files = sorted(glob.glob(os.path.join(folder, "*.xlsx")))

for fpath in files:
    print(f"\n{'='*60}")
    print(f"FILE: {os.path.basename(fpath)}")
    print(f"{'='*60}")
    wb = openpyxl.load_workbook(fpath)
    for shname in wb.sheetnames:
        ws = wb[shname]
        print(f"\n  Sheet: {shname}")
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            print("    (empty)")
            continue
        # Print header
        print(f"    Header: {rows[0]}")
        if len(rows) <= 1:
            continue
        # Find the class/category column
        header = [str(h).lower() if h else '' for h in rows[0]]
        print(f"    Total data rows: {len(rows)-1}")
        # Print ALL data rows
        for i, row in enumerate(rows[1:], 1):
            print(f"    {i}: {row}")
