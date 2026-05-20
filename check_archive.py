import os
from collections import Counter

# Check archive txt files - what class IDs appear as BACKGROUND on bus/truck images?
# If COCO IDs: car=2, motorcycle=3, bus=5, truck=7
# If Traffic14 IDs: personal_car=3, motorcycle=2, bus_solo=8/9, trucks=4/5/6/7

archive_base = r"C:\Users\admin\Trafic_mojo_2\Arhív\Training_pictures_reviewed"

print("=== Archive: class ID distribution per folder ===")
print("(checking what IDs appear in background annotations)")
print()

for folder in sorted(os.listdir(archive_base)):
    folder_path = os.path.join(archive_base, folder)
    if not os.path.isdir(folder_path):
        continue
    
    all_ids = Counter()
    multi_count = 0
    total_files = 0
    
    for fname in os.listdir(folder_path):
        if not fname.endswith('.txt'):
            continue
        total_files += 1
        txt_path = os.path.join(folder_path, fname)
        try:
            with open(txt_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) > 1:
                multi_count += 1
            for line in lines:
                parts = line.split()
                if parts:
                    all_ids[int(parts[0])] += 1
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
    
    if total_files == 0:
        continue
    
    ids_str = ", ".join(f"cls{k}:{v}" for k,v in sorted(all_ids.items()))
    print(f"  {folder} ({total_files} files, {multi_count} multi):")
    print(f"    IDs: {ids_str}")
    print()
