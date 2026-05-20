# Prompt: Desktop feldolgozó program frissítése 14 osztályos YOLO modellre

## Kontextus

A **Traffic Mojo** forgalomszámlálós rendszer desktop feldolgozó modulját (Flask + Python) kell
frissíteni. Az alkalmazás eddig egy COCO-alapú YOLO modellel működött (5 kategória: car, motorcycle,
bus, truck, person). Elkészült egy új, 14 osztályos egyedi YOLO11s modell, amelynek osztályai
teljesen eltérnek a COCO ID-któl.

**Fontos megkötés:** A tréningadat-gyűjtés funkcióját (`save_training_data`, `collect_classes`)
**NEM kell módosítani és NEM kell megtartani** a programban. Ez a funkció ki lesz kapcsolva / le lesz
távolítva, mert az adatgyűjtés egy külön YOLO-training modulban történik. A desktop feldolgozó
csak számol és mér, nem gyűjt tanítási adatot.

## Az új modell 14 osztálya (ID → neve)

```
0: person              (gyalogos)
1: bicycle             (kerékpár)
2: motorcycle          (motorkerékpár, robogó)
3: personal_car        (személyautó)
4: light_truck         (kisteherautó, furgon, dobozos < 3.5t)
5: medium_truck        (közepes teherjármű 3.5–12t)
6: heavy_truck         (nehéz teherjármű > 12t, merev)
7: vehicle_combination (nyerges vontatós / pótkocsis szerelvény)
8: bus_solo            (szóló nagybusz)
9: bus_articulated     (csuklós autóbusz)
10: trolley_solo       (szóló trolibusz)
11: trolley_articulated (csuklós trolibusz)
12: tram               (villamos)
13: minibus            (mikrobusz / kisbusz 9–20 fős)
```

## A módosítandó fájlok és konkrét változtatások

---

### 1. `config.py`

**Keresendő és módosítandó sorok:**
```python
classes_veh = [2, 3, 5, 7]
classes_ped = [0, 1]
active_classes = [2, 3, 5, 7]
```

**Csere erre:**
```python
classes_veh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # minden jármű
classes_ped = [0]                                              # csak gyalogos
active_classes = [2, 3, 4, 5, 6, 7, 8, 9, 13]  # motor, személyautó, teherek, buszok, minibus
```

A `collect_classes` sort töröld vagy kommentezd ki – a logikából ki lesz hagyva.

---

### 2. `process_video.py`

**Keresendő rész (~590. sor környékén, a kordon-metszési logikában):**
```python
if final_cls in [2, 3, 5, 7, 1]:
    possible_sets.append(veh_lines)
if final_cls in [0, 1, 3] and ped_lines not in possible_sets:
    possible_sets.append(ped_lines)
```

**Csere erre:**
```python
# Járműnek számít: bicycle(1), motorcycle(2), personal_car(3), light_truck(4),
# medium_truck(5), heavy_truck(6), vehicle_combination(7), bus_solo(8),
# bus_articulated(9), trolley_solo(10), trolley_articulated(11), tram(12), minibus(13)
if final_cls in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    possible_sets.append(veh_lines)
# Gyalogos kordonhoz: person(0) és bicycle(1)
if final_cls in [0, 1] and ped_lines not in possible_sets:
    possible_sets.append(ped_lines)
```

Az adatgyűjtési blokk eltávolítása (opcionális, de ajánlott a teljesítmény miatt):
Keresendő és törlendő kb. 15 sor, ahol `clean_frame` másolatot készít (`frame.copy()`)
és a track_info-ban `best_frame`, `best_box`, `best_snap_class` értékeket tárol.
Töröld a `save_training_snapshot` hívást is.

---

### 3. `gui_setup.py`

#### 3a. Figyelt kategóriák checkboxok

**Keresendő rész (`_build_ui` metódusban):**
```python
self.tracking_options = {"Gyalogos":0, "Kerékpár":1, "Autó":2, "Motor":3, "Busz":5, "Teher":7}
self.tracking_vars = {}

saved_active = getattr(Config, 'active_classes', [2,3,5,7])

for name, cid in self.tracking_options.items():
    var = tk.BooleanVar(value=cid in [2,3,5,7])
    self.tracking_vars[cid] = var
    ttk.Checkbutton(f, text=name, variable=var).pack(side="left", padx=10)
```

**Csere erre:**
```python
self.tracking_options = {
    "Gyalogos":      0,
    "Kerékpár":      1,
    "Motor":         2,
    "Személyautó":   3,
    "Kisteher":      4,
    "Középteher":    5,
    "Nehézteher":    6,
    "Szerelvény":    7,
    "Nagybusz":      8,
    "Csuklósbusz":   9,
    "Troli":        10,
    "Csuklóstroli": 11,
    "Villamos":     12,
    "Minibusz":     13,
}
self.tracking_vars = {}

saved_active = getattr(Config, 'active_classes', [2, 3, 4, 5, 6, 7, 8, 9, 13])

row_frame = ttk.Frame(f)
row_frame.pack(fill="x")
col = 0
for name, cid in self.tracking_options.items():
    var = tk.BooleanVar(value=cid in saved_active)
    self.tracking_vars[cid] = var
    cb = ttk.Checkbutton(row_frame, text=name, variable=var)
    cb.grid(row=col // 7, column=col % 7, sticky="w", padx=6, pady=2)
    col += 1
```

#### 3b. Adatgyűjtési szekció eltávolítása

A `_build_ui` metódusban töröld a teljes **„Adatgyűjtés tanításhoz"** LabelFrame blokkot
(`save_train_var`, `train_dir_var`, `class_options`, `class_vars`, a checkboxok).
Töröld a `toggle_train_ui` és `browse_train_dir` metódusokat is, valamint az
`__init__`-ben lévő `self.save_train_var` és `self.train_dir_var` inicializálásokat.

A `start_program` metódus visszaadott dict-jéből töröld a
`save_training_data`, `training_data_dir`, `collect_classes` kulcsokat.

#### 3c. Ablakmagasság

```python
self.root.geometry("750x850")
```
Módosítsd:
```python
self.root.geometry("750x900")
```

---

### 4. `templates/desktop.html`

#### 4a. Aktív objektum kategóriák checkboxok

**Jelenlegi 6 sor** (körülbelül ott ahol `cfg.active_classes` szerepel):
```html
<label class="class-item"><input type="checkbox" value="0"  {% if 0  in cfg.active_classes %}checked{% endif %}> 🚶 Gyalogos (0)</label>
<label class="class-item"><input type="checkbox" value="1"  {% if 1  in cfg.active_classes %}checked{% endif %}> 🚲 Kerékpár (1)</label>
<label class="class-item"><input type="checkbox" value="2"  {% if 2  in cfg.active_classes %}checked{% endif %}> 🚗 Személyautó (2)</label>
<label class="class-item"><input type="checkbox" value="3"  {% if 3  in cfg.active_classes %}checked{% endif %}> 🛵 Motorkerékpár (3)</label>
<label class="class-item"><input type="checkbox" value="5"  {% if 5  in cfg.active_classes %}checked{% endif %}> 🚌 Busz (5)</label>
<label class="class-item"><input type="checkbox" value="7"  {% if 7  in cfg.active_classes %}checked{% endif %}> 🚚 Teherjármű (7)</label>
```

**Csere erre (14 sor):**
```html
<label class="class-item"><input type="checkbox" value="0"  {% if 0  in cfg.active_classes %}checked{% endif %}> 🚶 Gyalogos (0)</label>
<label class="class-item"><input type="checkbox" value="1"  {% if 1  in cfg.active_classes %}checked{% endif %}> 🚲 Kerékpár (1)</label>
<label class="class-item"><input type="checkbox" value="2"  {% if 2  in cfg.active_classes %}checked{% endif %}> 🛵 Motor (2)</label>
<label class="class-item"><input type="checkbox" value="3"  {% if 3  in cfg.active_classes %}checked{% endif %}> 🚗 Személyautó (3)</label>
<label class="class-item"><input type="checkbox" value="4"  {% if 4  in cfg.active_classes %}checked{% endif %}> 🚐 Kisteherautó (4)</label>
<label class="class-item"><input type="checkbox" value="5"  {% if 5  in cfg.active_classes %}checked{% endif %}> 🚛 Középteher (5)</label>
<label class="class-item"><input type="checkbox" value="6"  {% if 6  in cfg.active_classes %}checked{% endif %}> 🚚 Nehézteher (6)</label>
<label class="class-item"><input type="checkbox" value="7"  {% if 7  in cfg.active_classes %}checked{% endif %}> 🚜 Szerelvény (7)</label>
<label class="class-item"><input type="checkbox" value="8"  {% if 8  in cfg.active_classes %}checked{% endif %}> 🚌 Nagybusz (8)</label>
<label class="class-item"><input type="checkbox" value="9"  {% if 9  in cfg.active_classes %}checked{% endif %}> 🚌 Csuklósbusz (9)</label>
<label class="class-item"><input type="checkbox" value="10" {% if 10 in cfg.active_classes %}checked{% endif %}> 🚎 Troli (10)</label>
<label class="class-item"><input type="checkbox" value="11" {% if 11 in cfg.active_classes %}checked{% endif %}> 🚎 Csuklóstroli (11)</label>
<label class="class-item"><input type="checkbox" value="12" {% if 12 in cfg.active_classes %}checked{% endif %}> 🚃 Villamos (12)</label>
<label class="class-item"><input type="checkbox" value="13" {% if 13 in cfg.active_classes %}checked{% endif %}> 🚐 Minibusz (13)</label>
```

#### 4b. Adatgyűjtési szekció eltávolítása

Töröld a teljes **„Gyűjtendő osztályok"** `<div>` blokkot (ahol `collect-item` class checkboxok vannak).
Ha az egész „Adatgyűjtés tanításhoz" config szekciót is el akarod távolítani, töröld az egész
section blokkot a `save_training_data` checkbox és `trainingDataDir` mező körül (~15-20 sor).

#### 4c. JavaScript `isVeh` függvény

**Jelenlegi:**
```javascript
const isVeh = cat => ['car','motorcycle','bus','truck','van'].includes(cat);
```

**Csere erre:**
```javascript
const isVeh = cat => ['motorcycle','personal_car','light_truck','medium_truck','heavy_truck',
    'vehicle_combination','bus_solo','bus_articulated','trolley_solo','trolley_articulated',
    'tram','minibus','bicycle'].includes(cat);
```

---

## Összefoglaló: régi COCO → új modell osztály-megfeleltetés

| Régi COCO név | Régi ID | Új osztálynév       | Új ID |
|---------------|---------|---------------------|-------|
| person        | 0       | person              | 0     |
| bicycle       | 1       | bicycle             | 1     |
| motorcycle    | 3       | motorcycle          | 2     |
| car           | 2       | personal_car        | 3     |
| truck (kis)   | 7       | light_truck         | 4     |
| truck (közép) | 7       | medium_truck        | 5     |
| truck (nehéz) | 7       | heavy_truck         | 6     |
| truck (szerel)| 7       | vehicle_combination | 7     |
| bus (szóló)   | 5       | bus_solo            | 8     |
| bus (csuklós) | 5       | bus_articulated     | 9     |
| –             | –       | trolley_solo        | 10    |
| –             | –       | trolley_articulated | 11    |
| –             | –       | tram                | 12    |
| –             | –       | minibus             | 13    |

## Fontos megjegyzések

1. **Ne változtass a kordonrajzolón, a sebességmérésen, a Google Sheets küldésen és a DataManager
   logikán** – csak a hardkódolt COCO class ID-ket és a UI-t kell frissíteni.
2. **A modell maga adja a neveket** – `model.names[final_cls]` automatikusan az új nevet adja
   (pl. `"personal_car"`), ezt sehol nem kell külön kezelni.
3. **`active_classes` integer lista** – a `.track(classes=active_classes)` hívás integer ID-kat
   vár, ez változatlan marad.
4. **Az adatgyűjtési kódot távolítsd el** – a `save_training`, `clean_frame`, `collect_set`,
   `best_frame`, `best_box`, `best_snap_class` változókat és a rájuk épülő logikát el kell távolítani
   minden fájlból.
