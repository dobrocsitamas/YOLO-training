from ultralytics import YOLO

model = YOLO("runs/train/traffic_v2/weights/best.pt")

# ── ONNX export ─────────────────────────────────────────────────────────────
model.export(format="onnx", imgsz=640, simplify=True)

# ── TensorRT export – Jetson Nano (FP16) ────────────────────────────────────
# Jetson Nano-n futtatandó; a parancsot a Jetson eszközön kell végrehajtani,
# mert a TensorRT engine platform-specifikus.
# model.export(format="engine", imgsz=640, half=True)   # FP16

# ── TensorRT export – Jetson Nano (INT8, kalibrálással) ─────────────────────
# INT8 gyorsabb, de kalibrációs adathalmazt igényel.
# model.export(format="engine", imgsz=640, int8=True, data="dataset/data.yaml")

# ── TorchScript export (opcionális) ─────────────────────────────────────────
# model.export(format="torchscript")
