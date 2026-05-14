from ultralytics import YOLO

# ── Modell betöltése (pretrained) ──────────────────────────────────────────
# Javasolt: yolo11s vagy yolo11m a 9 osztályhoz; n = gyorsabb, de kevésbé pontos
model = YOLO("yolo11s.pt")

# ── Tréning ────────────────────────────────────────────────────────────────
# Az adathalmaz elérési útja a prepare_dataset.py által generált data.yaml
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs/train",
    name="traffic_v2",
    device=0,          # 0 = GPU; "cpu" ha nincs GPU
    workers=4,
    patience=20,       # early stopping
)

print("Tréning kész:", results.save_dir)
