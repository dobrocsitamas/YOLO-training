from ultralytics import YOLO

# ── Modell betöltése (pretrained) ──────────────────────────────────────────
model = YOLO("yolo11n.pt")   # nano = legkisebb; l/x = nagyobb/pontosabb

# ── Tréning ────────────────────────────────────────────────────────────────
results = model.train(
    data="configs/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs/train",
    name="traffic_v1",
    device=0,          # 0 = GPU; "cpu" ha nincs GPU
    workers=4,
    patience=20,       # early stopping
)

print("Tréning kész:", results.save_dir)
