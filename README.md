# YOLO Training — Traffic Mojo

Ultralytics YOLO11 tréning projekt közlekedési kamera videókból.

## Struktúra

```
configs/         # dataset.yaml és modell konfig
data/
  images/train   # Roboflow exportból (gitignore-ban!)
  images/val
  labels/train
  labels/val
scripts/
  train.py       # tréning indítása
  validate.py    # kiértékelés
  export.py      # ONNX / TorchScript export
runs/            # Ultralytics kimenetek (gitignore-ban!)
```

## Gyors start

```bash
pip install -r requirements.txt

# 1. Roboflow-ból töltsd le a datasetet YOLO formátumban a data/ mappába
# 2. Szerkeszd a configs/dataset.yaml fájlt (osztályok, útvonalak)
# 3. Tréning indítása:
python scripts/train.py

# 4. Exportálás deployment-hez:
python scripts/export.py
```

## Roboflow integráció

A Roboflow projekt oldalán: **Export Dataset → YOLO v8 → show download code** —  
másold be a letöltési kódot és futtasd, a `data/` mappába kerüljön.
