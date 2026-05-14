from ultralytics import YOLO

model = YOLO("runs/train/traffic_v1/weights/best.pt")

# ONNX export (Traffic Mojo-hoz / deployment-hez)
model.export(format="onnx", imgsz=640, simplify=True)

# TorchScript export (opcionális)
# model.export(format="torchscript")
