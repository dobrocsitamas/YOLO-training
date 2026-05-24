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

## 2026-05-16–17 – Dataset elkészítése, tréning hibák javítása, tréning elindítása

### Dataset összeállítás

`prepare_dataset.py` **teljes újraírása** 14 osztályos sémára:
- Régi verzió placeholder annotációkat írt (`0 0.5 0.5 1.0 1.0`) minden képhez
- Új verzió a tényleges `.txt` annotációkat olvassa be, üres / hiányzó annotációjú képeket kihagyja
- COCO-kapcsolódó kód (`build_coco_id_remap`, `--coco_dir`) teljesen eltávolítva
- Futtatás: `python scripts/prepare_dataset.py --custom_root "C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed"`

**Eredmény:** 9272 kép → 7417 train / 1855 val

**Adat-tisztítási lépés:** 6 label fájlban érvénytelen (≥14) osztály ID-k maradtak
(régi COCO ID-k: 24, 26, 30). Python egysoros scripttel eltávolítva:
```python
import glob, pathlib
for f in glob.glob('dataset/labels/**/*.txt', recursive=True):
    lines = [l for l in pathlib.Path(f).read_text().splitlines() if l and int(l.split()[0]) < 14]
    pathlib.Path(f).write_text('\n'.join(lines) + '\n')
```

### Tréning indítási hibák javítása

**1. hiba – Windows multiprocessing crash:**
- `RuntimeError: freeze_support()` – a `train.py` hiányzó `if __name__ == '__main__':` guard miatt
  a worker processek rekurzívan reimportálták a fő modult
- Javítás: teljes tréning kód becsomagolva `if __name__ == '__main__':` blokk alá

**2. hiba – OSError: page file too small:**
- `workers=4` esetén a CUDA DLL-ek nem tölthetők be a subprocess-ekben Windows
  virtuális memória korlát miatt
- Javítás: `workers=0` (single-process dataloader) – ~3–4 perc/epoch vs. ~2 perc,
  de stabilan fut

### Tréning paraméterek

```python
model = YOLO("yolo11s.pt")
model.train(
    data="dataset/data.yaml",
    epochs=100, imgsz=640, batch=16,
    device=0, workers=0, patience=20,
    project="runs/train", name="traffic_v2",
    degrees=10.0, scale=0.5, shear=2.0,
    fliplr=0.5, flipud=0.0,
    mosaic=1.0, mixup=0.15, copy_paste=0.1,
)
```

Futtatás mappája: `runs/train/traffic_v2-4/`

---

## 2026-05-17–18 – Tréning befejezése, kiértékelés

### Tréning eredmény

- **85 epoch** futott le (early stopping, patience=20, legjobb: epoch 66)
- Teljes futásidő: ~8.4 óra
- Modell: `runs/detect/runs/train/traffic_v2-4/weights/best.pt`

**Összesített metrikák (best epoch 66):**

| Metrika | Érték |
|---------|-------|
| mAP50 | **0.710** |
| mAP50-95 | **0.484** |
| Precision | 0.703 |
| Recall | 0.686 |
| val cls_loss | 0.749 |

**Per-class validáció (best.pt):**

| Osztály | mAP50 | Értékelés |
|---------|-------|-----------|
| personal_car | 0.917 | ⭐ Kiváló |
| tram | 0.948 | ⭐ Kiváló (32 kép ellenére) |
| bus_articulated | 0.864 | ⭐ Kiváló |
| bus_solo | 0.845 | ⭐ Kiváló |
| person | 0.811 | Jó |
| bicycle | 0.692 | Közepes |
| light_truck | 0.700 | Közepes |
| heavy_truck | 0.675 | Közepes |
| vehicle_combination | 0.676 | Közepes |
| minibus | 0.619 | Gyenge – több adat kell |
| medium_truck | 0.611 | Gyenge – több adat kell |
| motorcycle | 0.577 | Gyenge – kevés adat (50 kép) |
| trolley_articulated | 0.249 | ❌ Gyakorlatilag nincs adat |
| trolley_solo | – | ❌ Nincs adat |

**Inference sebesség:** 1.6ms/kép (GPU) – ~625 FPS

### Következő tréning előtt elvégzendők

1. **Éles teszt** – számlálás feldolgozásban a `best.pt` modellel
2. **Adatgyűjtés** (prioritás sorrendben):
   - `motorcycle` – min. +150 kép
   - `medium_truck` – min. +200 kép
   - `minibus` – min. +150 kép
   - `trolley_solo` / `trolley_articulated` – bármennyi (helyszíni videó)
3. **Annotáció-minőség javítása** (labelImg vagy beépített szerkesztő) – különösen
   `medium_truck` és `minibus` osztályoknál rossz box-ok lehetnek
4. **Teljes újratréning** `yolo11s.pt` alapmodelltől – nem fine-tuning, mert az
   új adatoknak elejétől látni kell az egész datasetet
   
### Döntések / tanulságok

- `workers=0` marad Windows alatt – stabil, minimális sebesség-veszteség
- Fine-tuning a `best.pt`-ről **nem ajánlott** új adatok hozzáadásakor:
  catastrophic forgetting kockázata + alacsony LR miatt lassú tanulás
- A tram kiváló eredménye (0.948) mutatja: vizuálisan egyedi osztályhoz
  elegendő ~30-50 kép is, ha az annotáció jó minőségű
- mAP50 vs. mAP50-95 különbség (0.71 vs. 0.48) arra utal, hogy a bounding boxok
  pozíciója pontatlan egyes osztályoknál – annotáció-javítás javasolt

---

## 2026-05-20 – Dataset tisztítás + Traffic14_V3 tréning

### Előzmény: Traffic14_V2 éles teszt kudarca

Ugyanazon 10 perces tesztvideon (2026-04-16, Csepel Betű utca):
- YOLO11S: ~256 jármű detektálva
- Traffic14_V2: csak ~61 jármű – súlyos aluldetektálás

**Gyökérokok feltárása:**

1. **`review_annotations.py` ID-bug**: A teherautó alkategóriák (light/medium/heavy_truck,
   vehicle_combination) COCO ID-val kerültek tárolásra (pl. heavy_truck mappában
   ID=7 volt vehicle_combination helyett ID=6). A `filter_labels_to_trigger.py`
   szkript javított: 675 ID helyesen remappelve a kin-group logikával.

2. **Multi-annotáció zaj**: Minden képen az összes háttérjármű annotálva volt,
   de ezek soha nem lettek manuálisan ellenőrizve → szisztematikus félreclass-ifikációs
   zaj. Megoldás: **trigger-only módszer** – képenként csak az ellenőrzött fő jármű
   annotációja marad.

### Dataset újraépítés

**Szkriptek lefuttatva (ebben a sorrendben):**

1. `filter_labels_to_trigger.py` – csak trigger annotációk megtartása, ID-bug javítása
   - 675 ID remappelve
   - 94 fájl üres maradt (nincs trigger annotáció)

2. `check_and_delete_empty.py` – üres .txt + párolt .jpg törlése
   - Törölve: minibus:60, bus_solo:24, motorcycle:3, personal_car:3,
     bus_articulated:2, light_truck:1, tram:1 → összesen 94 pár

3. Régi dataset archiválva (9 GB, 9272 kép), majd töröve.
   Dataset újragenerálva `build_dataset.py`-vel.

**Végeredmény:**
- Train: 7342 kép, Val: 1836 kép
- Összesen: 14883 annotáció
- Osztályonkénti eloszlás:
  ```
  person:4184  bicycle:237  motorcycle:40  personal_car:4877
  light_truck:1674  medium_truck:703  heavy_truck:248
  vehicle_combination:1146  bus_solo:604  bus_articulated:710
  trolley_solo:0  trolley_articulated:0  tram:48  minibus:412
  ```

### Tréning: traffic_v2-5

```
Alap modell:  yolo11s.pt (scratch – nem fine-tuning)
epochs=100, imgsz=640, batch=16, patience=20
device=0, workers=0, mosaic=1.0, mixup=0.15, copy_paste=0.1
```

**Eredmény:**
- Futásidő: 8.849 óra (RTX 5060)
- Best epoch: 71 (early stop 91-nél)
- **mAP50 (all): 0.786** (vs. V2: 0.710, +10.7%)
- mAP50-95: 0.547

Osztályonkénti mAP50:
| Osztály              | mAP50 | Megjegyzés |
|----------------------|-------|------------|
| person               | 0.764 | |
| bicycle              | 0.812 | |
| motorcycle           | 0.982 | |
| personal_car         | 0.689 | diversity hiánya (csak Csepel) |
| light_truck          | 0.769 | |
| medium_truck         | 0.759 | |
| heavy_truck          | 0.461 | ❌ kevés adat (248 kép) |
| vehicle_combination  | 0.840 | |
| bus_solo             | 0.879 | |
| bus_articulated      | 0.877 | |
| trolley_solo         | 0.000 | ❌ nincs tanítókép |
| trolley_articulated  | 0.000 | ❌ nincs tanítókép |
| tram                 | 0.895 | |
| minibus              | 0.711 | |

### Traffic14_V3 csomagolás

Modellfájl: `TM_modulok_py/Traffic_Mojo_2_0/YOLO_models/Traffic14_V3/`
- `traffic14_v3.pt` – best.pt másolata (19.2 MB)
- `traffic14_v3.categories` – JSON osztályleíró (14 osztály, magyar nevek)
- `model.info` – teljesítményadatok, metodológia

### Következő lépések

1. **Éles teszt** – Traffic14_V3 ugyanazon tesztvideon (cél: >200 detektálás)
2. **`config_local.py` frissítése** → V3 útvonalra
3. **Képgyűjtő tooling fejlesztése** (időablakos mintavétel, minőségszűrők,
   osztályonkénti cap)
4. **Diversity javítás** – több helyszín, más kameraállások
5. **V4 tréning** az összegyűjtött változatosabb adattal

### Döntések / tanulságok

- **Trigger-only annotáció** bevált: tisztább osztályhatárok, nincs háttérzaj
- A vizuális diversity hiánya (csak Csepel, csak 151-es kék csuklós busz) a
  személyautó gyenge mAP50-jának fő oka – V4-ben ezt kell megoldani
- `heavy_truck` (0.461): adatmennyiség az elsődleges probléma, nem annotáció
- `nc=14` marad (trolleybus class-ok bennmaradnak, amíg adat nincs hozzájuk)

---

## 2026-05-20 – Traffic14_V3 éles teszt: kudarc + stratégiaváltás

### V3 éles teszt eredménye

Ugyanaz a 10 perces tesztvideon (2026-04-16, Csepel Betű utca):
- YOLO11S:       ~256 jármű
- Traffic14_V2:  ~61 jármű
- Traffic14_V3:  ~19 jármű (conf=0.25 és conf=0.10 mellett is!) – nem javult

**Gyökérok:** A `filter_labels_to_trigger.py` minden más osztály képéről törölte
a személyautó/gyalogos háttér-annotációkat. Eredmény: ~7700 képen a személyautó
jelöletlen → modell megtanulta: „háttérben lévő autó = háttér". val mAP50=0.786
félrevezető volt – a validáció is trigger-only adatból állt.

### Annotáció-minőség vizsgálat (browse_annotations.py)

Új eszköz: `scripts/browse_annotations.py` – tkinter böngésző bounding boxokkal,
Traffic14/COCO schema-váltóval, tetszőleges mappára navigálással (📁 Tallózás gomb),
archív összehasonlítással.

Feltárt problémák az archív annotációkban:
1. **COCO→Traffic14 remapping hiba**: cls5(bus)→cls5(medium_truck!),
   cls7(truck)→cls7(vehicle_combination!) – háttér bus/truck annotációk rossz kategóriát kaptak
2. **Óriási bounding boxok**: közeli járműveknél w>0.5, h>0.5
3. **Hiányzó háttér-annotációk**: filter_labels_to_trigger.py törölte őket

**Megbízható háttér-annotációk:** csak cls0/1/2/3 (person/bicycle/motorcycle/car).
**Megbízhatatlan:** cls4-13 háttér-annotációk.

### Stratégiaváltás: kétlépéses (two-stage) architektúra ← ÚJ IRÁNY

**Felismerés:** A YOLO11S detektálása tökéletes (person/car/truck/bus szinten).
Nem kell új YOLO modell – csak másodlagos classifier a truck/bus alkategóriákhoz.

```
YOLO11S (változatlan, tökéletes detektálás)
  → "truck" találat + kordon-átlépés
      → truck_classifier: light / medium / heavy / vehicle_combination
  → "bus" találat + kordon-átlépés
      → bus_classifier: solo / articulated
```

**Tanítóadatok már megvannak** (`Training_pictures_reviewed/`):
- Truck: light(1598) + medium(655) + heavy(227) + vehicle_combination(1022) = 3502 kép
- Bus: solo(627) + articulated(720) = 1347 kép

**Integráció:** Kordon-átlépéskor `process_video.py` kivágja a track legjobb
frame-jét (legjobb confidence), átadja a classifier-nek, finomított kategória kerül az Excel-be.

### Következő lépések (holnap)

1. Truck classifier tanítása (EfficientNet-B0 / ResNet18, 4 osztály, ~3500 kép)
2. Bus classifier tanítása (2 osztály, ~1347 kép)
3. Integráció `process_video.py`-ba: best-frame kivágás + classifier hívás
4. Éles teszt

---

## 2026-05-21 – Classifier integráció + virtuális ID rendszer

### Elkészült: `vehicle_classifier.py` modul
- `VehicleClassifier` osztály: `load(dir)`, `predict(crop)`, `display_name(clf_result)`, `is_applicable(yolo_cls)`
- `scan_classifier_models(base_dir)` → `[{path, display}]` lista a web UI dropdown-hoz
- EfficientNet-B0 alapú, `classifier.categories` JSON metaadat fájl alapján tölt be

### Elkészült: `classifier_models/vehicle_v1/`
- Architektúra: EfficientNet-B0, input: 224×224, margin: 10%
- Val accuracy: **95.2%**
- `applies_to_yolo_classes: [5, 7]` (COCO: busz, tehergépkocsi)
- Output osztályok: `light_truck`, `heavy_truck`, `vehicle_combination`, `bus_solo`, `bus_articulated`
- Tanítási minták: light_truck=923, heavy_truck=343, vehicle_combination=406, bus_solo=216, bus_articulated=84

### Megoldott: COCO ID ütközés → virtuális ID rendszer

**Probléma:** A YOLO11s standard COCO80 modell. Az `active_classes` lista közvetlenül megy a
`model.track(classes=...)` paraméterbe – tehát soha nem tartalmazhat 0–79-en kívüli vagy nem COCO
osztályt (pl. 4=airplane, 6=train stb. már foglalt).

**Megoldás:** 5xx/7xx prefix rendszer (szülő COCO ID × 100 + sorszám):

| Virtual ID | Jelentés | Szülő COCO ID |
|-----------|----------|---------------|
| 501 | Szóló busz | 5 |
| 502 | Csuklós autóbusz | 5 |
| 701 | Könnyű tehergépkocsi | 7 |
| 702 | Nehéz tehergépkocsi | 7 |
| 703 | Jármű szerelvény | 7 |

Virtual ID-k soha nem kerülnek a YOLO-ba – `_active_to_yolo_classes()` visszafordítja COCO ID-ra.

### Változtatások: `process_video.py`
- `_CLF_VIRTUAL_TO_YOLO` – virtual→COCO mapping szótár
- `_CLF_OUTPUT_TO_VIRTUAL` – classifier output string → virtual ID
- `_CLF_VIRTUAL_NAMES` – virtual ID → magyar megjelenítési név
- `_active_to_yolo_classes(active_classes)` – YOLO filter generálása (virtual ID-kat visszafejti COCO-ra)
- `_get_effective_cls(final_cls, obj_id, vehicle_clf, track_info)` – YOLO cls → virtual ID, cache-eli `track_info[obj_id]['clf_virtual_id']`-ben
- `model.track(classes=_active_to_yolo_classes(active_classes))` – YOLO soha nem kap virtual ID-t
- Counting: `eff_cls not in active_classes` → skip (szűrő, ha pl. csak bizonyos alosztályra kell)
- `matched_row` label: `_CLF_VIRTUAL_NAMES.get(eff_cls, yolo_fallback)` – pl. "Nehéz tehergépkocsi"

### Változtatások: `templates/desktop.html`
- `_collectActiveClasses()` helper: `data-ids="501,502"` típusú csoportos checkboxokat is kiteríti
- `renderClassCheckboxes` clf=ON: 5 egyedi COCO checkbox (0,1,2,3,12) + 2 csoportos gomb:
  - `data-ids="501,502"` → 🚌 Busz (szóló + csuklós együtt)
  - `data-ids="701,702,703"` → 🚚 Tehergépkocsi (könnyű / nehéz / szerelvény együtt)
- `onClfToggle`: automatikus ID-mapping váltáskor (5↔501+502, 7↔701+702+703)
- `saveConfig` + start payload: `_collectActiveClasses()` mindkét helyen

### Változtatások: `config.py` / `config_local.py`
- `config.py` default: `active_classes = [0,1,2,3,5,7,12]` (clf=OFF, tiszta COCO ID-k)
- `config_local.py` (jelen gép): `active_classes = [0,1,2,3,12,501,502,701,702,703]` (clf=BE)

---

## 2026-05-24 – Classifier hibák javítása + rezervoár mintavételezés

### Javított bug: `best_crop` soha nem töltődött be (process_video.py)

**Gyökérok:** A `track_info` inicializálásakor `'max_conf': current_conf` volt beállítva
(az első detektálás confidence értéke). Emiatt az `if current_conf > max_conf:` feltétel
az első frame-en soha nem teljesült (pl. `0.75 > 0.75` = False), tehát a `best_crop`
örökre `None` maradt azoknál a járműveknél, ahol az első detektálás volt a legmagasabb
confidence-ű. A classifier ennek következtében sohasem futott – a jármű nyers YOLO
`truck` kategóriaként kerülhetett a statisztikába, vagy teljesen kihagyódott.

**Miért érintette ez főleg a jármű szerelvényeket:** hosszú járműveknél az első megjelenés
pillanatában látható a teljes jármű a legtisztábban → ott a legmagasabb a YOLO confidence.
Később ahogy halad, a kép elvágódhat, a confidence csökken.

**Javítás:** `'max_conf': 0.0` → az első frame detektálása mindig elmenti a `best_crop`-ot.

### Továbbfejlesztés: rezervoár mintavételezés + batch predikció

**Probléma a javítás után:** A legjobb confidence-ű frame (jellemzően az első, frontális
nézet) nem alkalmas szerelvény-felismerésre – az oldalnézet a legmegbízhatóbb, de ott
már alacsonyabb a confidence.

**Megoldás:** Reservoir sampling algoritmus + batch inferencia:
- `clf_crops: []` – max `_MAX_CLF_CROPS = 5` kép gyűjtése **minden frame-ről** (nem csak
  a legjobb confidence-ű frame-ről), véletlenszerűen elosztva az egész track időtartamán
  (Algorithm R szerinti rezervoár csere: `j = random.randint(0, fc-1); if j < 5: csere`)
- A crop-gyűjtés **független a confidence értékétől** → garantáltan kerülnek közép-/
  oldalnézeti képek is a mintába
- `predict_batch(crops)` – az 5 képet egyetlen GPU forward pass-ban dolgozza fel
  (nem 5× lassabb, batch-párhuzamosítás)
- `softmax` valószínűségek alapján a **legmagasabb confidence-ű egyedi predikció** dönt
- A győztes crop visszakerül `best_crop`-ba a debug képmentéshez

### Új metódus: `vehicle_classifier.py` → `predict_batch(crops_bgr)`
- Input: `List[np.ndarray]` – BGR crop lista (None / üres crop-ok automatikusan kihagyva)
- Output: `[(class_name, confidence_float), ...]` – azonos sorrendben
- Belső: `torch.stack` → egyetlen forward pass → `softmax` → `max` per kép

### Változtatások összefoglalva

**`process_video.py`:**
- `import random` hozzáadva
- `_MAX_CLF_CROPS = 5` konstans
- `track_info` init: `max_conf: 0.0`, `clf_crops: []`, `clf_frame_count: 0`
- Crop-gyűjtés kikerült a `max_conf` blokkból → minden detektálásnál fut (is_applicable esetén)
- `_get_effective_cls()` újraírva: `predict_batch(crops)` hívás, legjobb prob döntés,
  győztes crop visszamentése `best_crop`-ba

**`vehicle_classifier.py`:**
- `predict_batch(crops_bgr)` új metódus

### Éles tesztek eredménye (2026-05-24, Csepel Betű utca)
- Szerelvény kategorizálás **érezhetően javult** a rezervoár mintavételezés után
- Korábban: szerelvények nagy része `Nehéz tehergépkocsi`-ként jelent meg → most helyes
- Közepes teher kategória: nincs (a modell 5 osztályra lett tanítva: light/heavy/vehicle_combination/bus_solo/bus_articulated)

### Hamis irány detektálás vizsgálata (v5→v7, v1→v4)
- `current_p = (int((box[0]+box[2])/2), int(box[3]))` – bounding box alsó szélének közepe –
  **egységes minden kategóriánál**, programhiba kizárva
- v5→v7: v5 jobb vége (935,331) és v7 bal vége (1017,243) mindössze ~82 px távolságra –
  hosszú jármű egyszerre metszheti mindkettőt
- v1→v4: v1 és v4 közel vannak a kép jobb oldalán – rövid úton átmegy rajta egyes jármű
- **Döntés:** kordon-elhelyezéssel oldja meg a felhasználó (kód változtatás nem szükséges)

---
