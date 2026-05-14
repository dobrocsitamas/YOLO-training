from ultralytics import YOLO

model = YOLO("runs/train/traffic_v1/weights/best.pt")

metrics = model.val(data="configs/dataset.yaml")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
