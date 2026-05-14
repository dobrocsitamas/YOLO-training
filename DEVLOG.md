# Fejlesztési napló

---

## 2026-05-14

### Elvégzett munka

**Cél:** 9 osztályos YOLO11 forgalomfigyelő modell tanítása saját videókból.

**Osztálystruktúra** (végleges):
| ID | Osztály |
|----|---------|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 4 | bus_solo |
| 5 | bus_articulated |
| 6 | truck_light |
| 7 | truck_heavy |
| 8 | vehicle_combination |

**Elkészült / módosított fájlok:**
- `scripts/extract_frames.py` – headless videóból frame kinyerés YOLO track()-kal, tqdm progress, konfigurálható `--triggers` (alapértelmezett: minden osztály)
- `scripts/review_annotations.py` – tkinter GUI bus/truck alkategóriák manuális besorolásához (billentyűk: 1-5, S=skip, ←=vissza)
- `scripts/prepare_dataset.py` – saját képekből YOLO dataset összeállítás (80/20 split, data.yaml generálás)
- `configs/dataset.yaml` – 9 osztályra frissítve
- `scripts/train.py` – modell: yolo11s.pt, név: traffic_v2
- `requirements.txt` – roboflow eltávolítva, tqdm hozzáadva
- `.gitignore` – data/, dataset/, training_data/, *.pt, *.onnx, *.engine blokkolva

**Eldobott megközelítés:** COCO 2017 + Roboflow adatbázis (osztály ID eltérések, nem megfelelő irány)

**Függőségek állapota:**
- Venv: `D:\YOLO training\.venv\Scripts\python.exe` (Python 3.12.10)
- torch 2.12.0, ultralytics 8.4.50, opencv-python, Pillow, tqdm – telepítve a venv-be
- ⚠️ PowerShell execution policy tiltja `.venv\Scripts\activate` futtatását

### Holnap folytatás

**1. Csomagok ellenőrzése:**
```powershell
cd "D:\YOLO training"
.venv\Scripts\python.exe -c "import tqdm, ultralytics, cv2; print('OK')"
```

**2. Ha execution policy hiba van (egyszer kell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**3. extract_frames.py futtatása valós videóval:**
```powershell
cd "D:\YOLO training"
.venv\Scripts\python.exe scripts/extract_frames.py `
  --video "D:/path/to/video.mp4" `
  --model "weights/best.pt" `
  --output training_data/raw `
  --device 0
```
- `--triggers bus,truck` – csak bus/truck gyűjtéshez
- `--triggers` elhagyva – minden osztályt gyűjt

**4. Manuális felülvizsgálat (bus/truck alkategóriák):**
```powershell
.venv\Scripts\python.exe scripts/review_annotations.py `
  --source training_data/raw `
  --output training_data/reviewed
```

**5. Dataset összeállítása:**
```powershell
.venv\Scripts\python.exe scripts/prepare_dataset.py `
  --custom_root training_data/reviewed
```

**6. Tanítás:**
```powershell
.venv\Scripts\python.exe scripts/train.py
```

**7. Export (Jetson Nano, TensorRT FP16):**
```powershell
.venv\Scripts\python.exe scripts/export.py
```

### Függőben lévő feladatok
- [ ] extract_frames.py tesztelése valós videóval + best.pt modellel
- [ ] Edzőadatok gyűjtése összes videóból
- [ ] review_annotations.py tesztelése
- [ ] prepare_dataset.py futtatása
- [ ] train.py futtatása
- [ ] Export TensorRT-re (Jetson Nano)

---
