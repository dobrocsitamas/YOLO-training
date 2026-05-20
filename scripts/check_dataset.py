import pathlib
from collections import Counter

labels_train = pathlib.Path(r"C:\Users\admin\Trafic_mojo_2\YOLO-training\dataset\labels\train")
CLASS_NAMES = ['person','bicycle','motorcycle','personal_car','light_truck','medium_truck','heavy_truck','vehicle_combination','bus_solo','bus_articulated','trolley_solo','trolley_articulated','tram','minibus']

counts = Counter()
empty = 0
for txt in labels_train.glob("*.txt"):
    content = txt.read_text().strip()
    if not content:
        empty += 1
        continue
    for line in content.splitlines():
        parts = line.strip().split()
        if parts:
            counts[int(parts[0])] += 1

print(f"Train label files: {len(list(labels_train.glob('*.txt')))}, empty: {empty}")
print("Class distribution:")
for cid, name in enumerate(CLASS_NAMES):
    print(f"  {cid} {name}: {counts.get(cid, 0)}")
print(f"Total annotations: {sum(counts.values())}")
