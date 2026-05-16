from ultralytics import YOLO

# ── Modell betöltése (pretrained) ──────────────────────────────────────────
# Javasolt: yolo11s vagy yolo11m a 13 osztályhoz; n = gyorsabb, de kevésbé pontos
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

    # ── Augmentáció ────────────────────────────────────────────────────────
    # Alapértelmezett értékek elegendők a nagy osztályokhoz (car, person).
    # Az alábbi beállítások a ritka osztályokat (bicycle, motorcycle, tram)
    # segítik: minden képből változatos tanítópéldákat generál, nem ment fájlt.

    hsv_h=0.015,       # szín-árnyalat jitter (alapért.: 0.015)
    hsv_s=0.7,         # telítettség jitter (alapért.: 0.7)
    hsv_v=0.4,         # fényerő jitter (alapért.: 0.4)
    degrees=10.0,      # forgatás ±10° – hasznos biciklinél/motornál
    translate=0.1,     # eltolás (alapért.: 0.1)
    scale=0.5,         # méretezés 50–150% (alapért.: 0.5)
    shear=2.0,         # nyírás ±2° – perspektíva-változás szimulálása
    perspective=0.0,   # perspektíva-torzítás (0 = kikapcsolva, max: 0.001)
    flipud=0.0,         # függőleges tükrözés – forgalmi kameránál nem reális
    fliplr=0.5,        # vízszintes tükrözés (alapért.: 0.5)
    mosaic=1.0,        # mozaik: 4 képet összevág egybe – kis objektumoknál sokat segít
    mixup=0.15,        # két kép keverése – ritka osztályok előfordulását növeli
    copy_paste=0.1,    # objektum kivágása és más képre illesztése – bicycle/motorcycle++
)

print("Tréning kész:", results.save_dir)
