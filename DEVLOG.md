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
- [ ] review_annotations.py futtatása a Training_pictures mappán
- [x] prepare_dataset.py ellenőrzése az új 13 osztályos mappastruktúrával
- [ ] Képgyűjtés folytatása – célzott gyűjtés hiányos kategóriákból
- [ ] train.py futtatása összegyűlt és felülvizsgált adaton
- [ ] Export TensorRT-re (Jetson Nano)

---

## 2026-05-16 – Adatminőség ellenőrzés, pipeline tisztázás, GUI fejlesztések

### Annotáció ID-k ellenőrzése

Megvizsgáltuk a `Training_pictures/` mappa `.txt` fájljait. Kiderült, hogy az
`extract_frames.py` **COCO osztály-ID-kat** ment a `.txt` fájlokba (pl. `car=2`,
`bus=5`, `truck=7`) – ez szándékos, köztes állapot. Az átindexelés a
`review_annotations.py` feladata (`remap_boxes()` függvény).

**A `training_data/` mappa régebbi formátumú adatokat tartalmaz** (csak 1 objektum /
kép, osztály mindig `0`). Ezeket egyelőre nem vonjuk be a tréningbe – a minőségi
különbség (egyobjektumos vs. teljes annotáció) miatt nem éri meg a konvertálás.
Archívként megmaradnak.

**Döntés:** a képgyűjtő funkciót kizárólag ebben a YOLO-training csomagban tartjuk.
A Desktop feldolgozó és a Jetson mérőprogramja nem foglalkozik tréningadat-gyűjtéssel
– ez gyorsítja a mérési algoritmust és egyetlen helyen tartja a logikát.

### Képállomány aktuális állapota (Training_pictures/)

| Kategória | Db    | Megjegyzés                              |
|-----------|-------|-----------------------------------------|
| car       | 12440 | ⚠️ Túl sok – review-nál max 2–3000-et megtartani |
| person    | 5463  | ✓ Elég                                  |
| truck     | 1971  | ⚡ Elegendő, de alkategóriánként kevés   |
| bus       | 1392  | ⚠️ Kell még ~1000 kép                   |
| bicycle   | 63    | 🔴 Kritikusan kevés                     |
| motorcycle| 10    | 🔴 Kritikusan kevés                     |
| tram      | 0     | 🔴 Hiányzik – `train` triggerrel gyűjthető |
| trolley   | 0     | ⏳ Később – helyszíni videó szükséges    |

### Kódváltozások

**`extract_frames_gui.py`**
- `ALL_CLASSES` listához hozzáadva a `train` osztály → villamost lehet gyűjteni
  a YOLO11 `train` (COCO ID: 6) detekciójára alapozva

**`review_annotations.py`**
- `FOLDER_MAP`-be felvéve a `train` mappa: `COCO 6 → kötelező manuális választás`
  (review-nál `R` = tram/12, `S` = kihagyás ha nem villamos)
- **Session számláló panel** hozzáadva a jobb oldali panelbe: futás közben
  osztályonként mutatja a jóváhagyott képek számát. Visszavonásnál helyesen
  csökkenti a számlálót is. Hasznos truck review-nál: látható mikor van már
  elég `light_truck` és mikor érdemes azt kihagyni (`S`).

**`start_gui.bat`** *(új fájl)*
- VS Code-on kívül is indítható launcher a `extract_frames_gui.py`-hoz
- Python keresési sorrend: venv → .venv → Conda → rendszer Python
- Hibaüzenet ha Python nem található vagy csomag hiányzik

**`start_gui.ico`** *(új fájl)*
- Egyedi ikon a launcherhez (film keret + play gomb + autó szilhouett)
- Desktop parancsikon (`Frame Extractor.lnk`) létrehozva az ikonnal

### Következő lépések
1. Célzott képgyűjtés új videókból: `motorcycle`, `bicycle`, `bus`, `truck`, `train`
2. `review_annotations.py` futtatása:
   ```powershell
   .venv\Scripts\python.exe scripts/review_annotations.py `
     --source "C:\Users\admin\Trafic_mojo_2\Training_pictures" `
     --output "C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed"
   ```
3. `prepare_dataset.py` futtatása a reviewed mappán
4. `train.py` – első tréning futtatás

---

## 2026-05-16 (folytatás) – Célzott képgyűjtés, augmentáció, 14 osztályos séma

### Második képgyűjtési kör – eredmény

Célzott gyűjtés prioritás szerint (motorcycle, bicycle, bus, truck, train).
Eredmény a `Training_pictures/` mappában:

| Kategória | Előző | Most   | Változás |
|-----------|-------|--------|----------|
| truck     | 1 971 | 7 469  | +5 498   |
| bus       | 1 392 | 2 879  | +1 487   |
| bicycle   | 63    | 226    | +163     |
| motorcycle| 10    | 59     | +49      |
| train     | 0     | 16     | +16 (új) |
| car       | 12 440| 12 440 | —        |
| person    | 5 463 | 5 463  | —        |

**Megjegyzések:**
- `motorcycle` és `bicycle` még mindig kevés, de augmentációval kezelhető
- `train` (villamos) 16 db – minimális, pótlás szükséges ha lesz helyszíni videó
- `car` 12 440 db – review-nál max 2–3000-et érdemes megtartani az arányok miatt
- Trolibusz kategória egyelőre nem kerül bele – helyszíni videó szükséges

### Adataugmentáció beállítása (`train.py`)

A YOLO beépített augmentációja fut tréning közben (nem ment fájlt, epoch-onként
újra generál). Fokozott beállítások a ritka osztályok (bicycle, motorcycle, tram)
miatt:

```python
degrees=10.0    # forgatás ±10°
scale=0.5       # méretezés 50–150%
shear=2.0       # nyírás ±2°
fliplr=0.5      # vízszintes tükrözés
flipud=0.0      # függőleges tükrözés – forgalmi kameránál nem reális
mosaic=1.0      # 4 képet összevág → kis objektumok többször jelennek meg
mixup=0.15      # két kép keverése → ritka osztályok előfordulása nő
copy_paste=0.1  # objektum kivágása és más képre illesztése → bicycle/motorcycle++
```

Hatás: bicycle/motorcycle esetén ~3–5× több effektív tanítópélda epoch-onként.

### 14 osztályos séma – minibus hozzáadva

**Indoklás:** a mikrobusz (Sprinter Bus, Transit Bus, 9–20 fős) forgalomtechnikai
szempontból különbözik a nagybusztól és a dobozos kisteherautótól. A YOLO modell
`bus`-ként detektálja, de review-nál szét kell választani.

**Osztályhatárok:**

| Jármű típus              | Osztály            | ID | Review billentyű |
|--------------------------|--------------------|----|-----------------|
| Dobozos furgon, Transit Cargo | `light_truck`  | 4  | `5`             |
| Mikrobusz, Sprinter Bus  | `minibus`          | 13 | `T` (új)        |
| Szóló nagybusz (>20 fős) | `bus_solo`         | 8  | `9`             |
| Csuklós busz             | `bus_articulated`  | 9  | `Q`             |

**Módosított fájlok:**

`configs/dataset.yaml`
- `nc: 13` → `nc: 14`
- `light_truck` leírás pontosítva: *furgon, dobozos kisteher (áruszállítás, <3.5t)*
- `bus_solo` leírás pontosítva: *nagybusz, szóló (>20 fős)*
- Új osztály: `13: minibus` – *mikrobusz/kisbusz (9–20 fős)*

`scripts/review_annotations.py`
- `CLASS_NAMES` listába felvéve: `"minibus"` (index 13)
- `KEY_MAP`-be felvéve: `"t": 13`
- `CLASS_COLORS`-ba felvéve: `13: "#f9e2af"` (sárga)
- Súgószöveg frissítve: `1-9 / Q,W,E,R,T → osztályok`
- Gombsor (`key_labels`) frissítve a `T` billentyűvel

### Következő lépések
1. `review_annotations.py` futtatása – bus és truck képek manuális osztályozása
   (figyelj a minibus ↔ light_truck ↔ bus_solo határokra!)
2. Review közben session-számláló segít nyomon követni az alkategóriák arányát
3. `prepare_dataset.py` futtatása a reviewed mappán
4. `train.py` – első tréning futtatás (14 osztály, augmentációval)

---
