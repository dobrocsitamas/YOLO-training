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

## 2026-05-15

### Elvégzett munka

**Cél:** Osztálystruktúra bővítése 9 → 13 osztályra, teljes manuális képfelülvizsgálat megvalósítása, inkrementális dataset-bővítés támogatása.

**Osztálystruktúra** (új, 13 osztály):
| ID | Osztály | Megjegyzés |
|----|---------|-----------|
| 0 | person | COCO alaposztály |
| 1 | bicycle | COCO alaposztály |
| 2 | motorcycle | COCO alaposztály (motorkerékpár, robogó) |
| 3 | personal_car | COCO car → átnevezve |
| 4 | light_truck | kisteherautó, furgon (< 3.5t) |
| 5 | medium_truck | közepes teherjármű (3.5–12t) |
| 6 | heavy_truck | nehéz teherjármű (> 12t, merev) |
| 7 | vehicle_combination | nyerges vontatós / pótkocsis szerelvény |
| 8 | bus_solo | szóló autóbusz |
| 9 | bus_articulated | csuklós autóbusz |
| 10 | trolley_solo | szóló trolibusz |
| 11 | trolley_articulated | csuklós trolibusz |
| 12 | tram | villamos |

**Elkészült / módosított fájlok:**

- `scripts/review_annotations.py` – **teljes újraírás**
  - Minden kategória megjelenik (nem csak bus/truck)
  - **Inkrementális logika:** a `Training_pictures_reviewed/` mappában már szereplő fájlneveket kihagyja → újabb videók feldolgozása után csak az új képeket mutatja
  - **Gyors jóváhagyás `[O]`**: egyszerű osztályoknál (person/bicycle/motorcycle/car) azonnal menti az alapértelmezett osztállyal
  - **Kötelező manuális választás**: bus/truck esetén az `[O]` le van tiltva
  - **13 osztály billentyűi**: `1–9` + `Q` (bus_articulated) + `W` (trolley_solo) + `E` (trolley_articulated) + `R` (tram)
  - **`[←]` visszavonás**: törli az utolsó kimenetbe mentett fájlokat, visszalép
  - **COCO ID → új ID remapping**: .txt annotációkban az osztály ID-k automatikusan frissülnek
  - Alapértelmezett útvonalak: forrás `Training_pictures/`, kimenet `Training_pictures_reviewed/`

- `configs/dataset.yaml` – frissítve 13 osztályra

**Munkafolyamat (inkrementális bővítés):**
1. `extract_frames_gui.py` → új képek gyűjtése `Training_pictures/`-be
2. `review_annotations.py` → **csak az új, még nem ellenőrzött képek** jelennek meg
3. `prepare_dataset.py --custom_root Training_pictures_reviewed` → dataset összerakás

**Környezet (asztali PC):**
- Python 3.11.9, venv: `C:\Users\admin\Trafic_mojo_2\YOLO-training\.venv`
- NVIDIA RTX 5060 (8GB VRAM), driver 595.79, CUDA 12.8
- torch 2.11.0+cu128, ultralytics 8.4.51, opencv-python 4.13.0, Pillow 12.2.0
- Modell: `C:\Users\admin\Trafic_mojo_2\TM_modulok_py\Traffic_Mojo_2_0\yolo11s.pt`
- Videók: `C:\Users\admin\Trafic_mojo_2\Video\` (3 nagy ~36GB + `Teszt videók\` 8 kisebb klip)
- Gyűjtött képek: `C:\Users\admin\Trafic_mojo_2\Training_pictures\`

**Futtatás:**
```powershell
cd "C:\Users\admin\Trafic_mojo_2\YOLO-training"
.venv\Scripts\python.exe scripts/review_annotations.py
# vagy egyedi útvonalakkal:
.venv\Scripts\python.exe scripts/review_annotations.py `
  --source "C:\Users\admin\Trafic_mojo_2\Training_pictures" `
  --output "C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed"
```

### Függőben lévő feladatok
- [ ] review_annotations.py tesztelése valós képekkel
- [ ] prepare_dataset.py ellenőrzése az új 13 osztályos mappastruktúrával
- [ ] Képgyűjtés folytatása a nagy videókból
- [ ] train.py futtatása összegyűlt és felülvizsgált adaton
- [ ] Export TensorRT-re (Jetson Nano)

---
