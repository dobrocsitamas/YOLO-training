from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/runs/train/traffic_v2-4/weights/best.pt")

    metrics = model.val(data="dataset/data.yaml", workers=0)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
