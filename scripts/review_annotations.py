#!/usr/bin/env python3
"""
Annotation Review Tool – 13 osztályos verzió
=============================================
Manuális annotáció-ellenőrző az összes gyűjtött képhez.

Minden kép manuális jóváhagyást igényel.
A már felülvizsgált képek kimaradnak a következő futtatásból –
így folyamatosan bővíthető a dataset új videók feldolgozásával.

Forrás mappastruktúra (extract_frames.py kimenete):
  <source>/
    person/     → person (0)              – O-val gyorsan jóváhagyható
    bicycle/    → bicycle (1)             – O-val gyorsan jóváhagyható
    motorcycle/ → motorcycle (2)          – O-val gyorsan jóváhagyható
    car/        → personal_car (3)        – O-val gyorsan jóváhagyható
    bus/        → kötelező kézi választás (bus_solo / bus_articulated / trolley_* / tram)
    truck/      → kötelező kézi választás (light_truck / medium_truck / heavy_truck / vehicle_combination)

Billentyűzet:
  O         → jóváhagyás az alapértelmezett osztállyal (csak egyszerű kategóriákra)
  1–9       → osztályok 0–8  (person … bus_solo)
  Q         → bus_articulated (9)
  W         → trolley_solo (10)
  E         → trolley_articulated (11)
  R         → tram (12)
  S         → kihagyás (skipped/ mappába)
  ←         → visszavonás / előző kép

Kimenet:
  <output>/
    person/ bicycle/ motorcycle/ personal_car/
    light_truck/ medium_truck/ heavy_truck/ vehicle_combination/
    bus_solo/ bus_articulated/ trolley_solo/ trolley_articulated/ tram/
    skipped/

A már kimenetben lévő képeket a program automatikusan kihagyja,
így bármikor futtatható újabb képgyűjtés után.

Következő lépés:
  python scripts/prepare_dataset.py --custom_root <output>

Használat:
  python scripts/review_annotations.py
  python scripts/review_annotations.py --source <forrásmappa> --output <kimeneti mappa>
"""

import argparse
import shutil
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError:
    print("[HIBA] Pillow nem telepítve. Futtasd: pip install Pillow")
    sys.exit(1)


# ── 13 osztályos struktúra ────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    "person",              # 0
    "bicycle",             # 1
    "motorcycle",          # 2
    "personal_car",        # 3
    "light_truck",         # 4
    "medium_truck",        # 5
    "heavy_truck",         # 6
    "vehicle_combination", # 7
    "bus_solo",            # 8
    "bus_articulated",     # 9
    "trolley_solo",        # 10
    "trolley_articulated", # 11
    "tram",                # 12
]

# Forrásmappa neve → (COCO class ID a .txt-ben, alapértelmezett új class ID vagy None)
# None = kötelező manuális választás (nem nyomható O)
FOLDER_MAP: dict[str, tuple[int, int | None]] = {
    "person":     (0, 0),    # COCO 0 → new 0 (person)
    "bicycle":    (1, 1),    # COCO 1 → new 1 (bicycle)
    "motorcycle": (3, 2),    # COCO 3 → new 2 (motorcycle)
    "car":        (2, 3),    # COCO 2 → new 3 (personal_car)
    "bus":        (5, None), # COCO 5 → kötelező választás
    "truck":      (7, None), # COCO 7 → kötelező választás
    "train":      (6, None), # COCO 6 → kötelező választás (villamos vagy nem villamos?)
}

# Gyorsbillentyűk → új class ID
KEY_MAP: dict[str, int] = {
    "1": 0,   # person
    "2": 1,   # bicycle
    "3": 2,   # motorcycle
    "4": 3,   # personal_car
    "5": 4,   # light_truck
    "6": 5,   # medium_truck
    "7": 6,   # heavy_truck
    "8": 7,   # vehicle_combination
    "9": 8,   # bus_solo
    "q": 9,   # bus_articulated
    "w": 10,  # trolley_solo
    "e": 11,  # trolley_articulated
    "r": 12,  # tram
}

# Új osztályok megjelenítési színei (new class ID → hex szín)
CLASS_COLORS: dict[int, str] = {
    0:  "#a6e3a1",  # person
    1:  "#89dceb",  # bicycle
    2:  "#89b4fa",  # motorcycle
    3:  "#b4befe",  # personal_car
    4:  "#a6e3a1",  # light_truck
    5:  "#fab387",  # medium_truck
    6:  "#f38ba8",  # heavy_truck
    7:  "#cba6f7",  # vehicle_combination
    8:  "#fe640b",  # bus_solo
    9:  "#e64553",  # bus_articulated
    10: "#04a5e5",  # trolley_solo
    11: "#209fb5",  # trolley_articulated
    12: "#7287fd",  # tram
}

# Forrás .txt fájlok COCO ID-jainak megjelenítési adatai
COCO_LABEL: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
}
COCO_COLOR: dict[int, str] = {
    0: "#a6e3a1",  # person
    1: "#89dceb",  # bicycle
    2: "#b4befe",  # car
    3: "#89b4fa",  # motorcycle
    5: "#fe640b",  # bus  (narancs – kiemelés)
    7: "#f38ba8",  # truck (piros – kiemelés)
}

MAX_CANVAS_W = 960
MAX_CANVAS_H = 620


# ── YOLO .txt segédfüggvények ─────────────────────────────────────────────────

def read_txt(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), float(parts[1]),
                               float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def write_txt(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remap_boxes(
    boxes: list[tuple[int, float, float, float, float]],
    old_coco_id: int,
    new_id: int,
) -> list[tuple[int, float, float, float, float]]:
    """A megadott COCO class ID-jű dobozokat átírja az új ID-ra.
    A többi ismert COCO ID-t is leképezi az új struktúrára."""
    COCO_TO_NEW: dict[int, int] = {0: 0, 1: 1, 2: 3, 3: 2}
    result = []
    for cls, cx, cy, w, h in boxes:
        if cls == old_coco_id:
            result.append((new_id, cx, cy, w, h))
        else:
            mapped = COCO_TO_NEW.get(cls, cls)
            result.append((mapped, cx, cy, w, h))
    return result


# ── Kép segédfüggvények ───────────────────────────────────────────────────────

def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def render_image(
    img_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
    highlight_coco_id: int | None,
) -> Image.Image:
    """Visszaad egy PIL képet a bbox-okkal.
    A highlight_coco_id-jű dobozok vastag kerettel, kiemelve jelennek meg."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    for cls, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        is_highlight = (cls == highlight_coco_id)
        color = COCO_COLOR.get(cls, CLASS_COLORS.get(cls, "#ffffff"))
        label = COCO_LABEL.get(cls, CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}")
        lw = 4 if is_highlight else 1

        if is_highlight:
            label = f"▶ {label} ◀"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        try:
            draw.text((x1 + 3, y1 + 2), label, fill=color)
        except Exception:
            pass

    return img


# ── Képlista összeállítása ────────────────────────────────────────────────────

def collect_images(
    source_dir: Path,
    output_dir: Path,
) -> list[tuple[Path, str, int | None, int]]:
    """
    Visszaadja a feldolgozandó képek listáját:
      (kép_path, forrásmappa_neve, alapért_new_id_vagy_None, coco_id)

    A már a kimeneti mappában lévő képeket kihagyja, így a program
    mindig csak az új, még nem felülvizsgált képeket mutatja.
    """
    already_done: set[str] = set()
    if output_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for f in output_dir.rglob(ext):
                already_done.add(f.name)

    result = []
    for folder_name, (coco_id, default_new_id) in FOLDER_MAP.items():
        folder = source_dir / folder_name
        if not folder.is_dir():
            continue
        images: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(folder.glob(ext))
        images.sort()
        for img_path in images:
            if img_path.name not in already_done:
                result.append((img_path, folder_name, default_new_id, coco_id))

    return result


# ── Fő alkalmazás ─────────────────────────────────────────────────────────────

class ReviewApp:
    def __init__(self, source_dir: Path, output_dir: Path) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self.images = collect_images(source_dir, output_dir)
        self.total = len(self.images)
        self.img_idx = 0

        # Visszavonás história: minden lépésnél a kimenetbe mentett fájlok listája
        self._history: list[tuple[Path, ...]] = []

        # Session-számlálók: osztálynév → jóváhagyott db száma ebben a futásban
        self._session_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
        self._session_skipped: int = 0

        self.root = tk.Tk()
        self.root.title("Annotáció Átnéző – 13 osztály")
        self.root.configure(bg="#1e1e2e")
        self._build_ui()
        self._bind_keys()
        self._load_current_image()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        canvas_frame = tk.Frame(self.root, bg="#181825")
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        self.canvas = tk.Canvas(
            canvas_frame, bg="#11111b",
            width=MAX_CANVAS_W, height=MAX_CANVAS_H,
            highlightthickness=0,
        )
        self.canvas.pack()

        right = tk.Frame(self.root, bg="#1e1e2e", width=220)
        right.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=8)
        right.grid_propagate(False)

        self.lbl_progress = tk.Label(
            right, text="0 / 0", bg="#1e1e2e", fg="#cdd6f4",
            font=("Segoe UI", 11, "bold"),
        )
        self.lbl_progress.pack(pady=(8, 0))

        self.lbl_filename = tk.Label(
            right, text="", bg="#1e1e2e", fg="#6c7086",
            font=("Consolas", 7), wraplength=210,
        )
        self.lbl_filename.pack()

        self.lbl_source = tk.Label(
            right, text="", bg="#1e1e2e", fg="#89dceb",
            font=("Segoe UI", 9, "italic"),
        )
        self.lbl_source.pack(pady=(4, 0))

        tk.Frame(right, bg="#45475a", height=1).pack(fill="x", pady=6)

        self.btn_approve = tk.Button(
            right, text="✓ Jóváhagyás  [O]",
            command=self._approve,
            bg="#a6e3a1", fg="#1e1e2e", relief="flat",
            pady=8, width=22,
            font=("Segoe UI", 9, "bold"),
            activebackground="#94d3b0",
        )
        self.btn_approve.pack(pady=(0, 4), fill="x", padx=4)

        tk.Frame(right, bg="#45475a", height=1).pack(fill="x", pady=4)

        self.btn_frame = tk.Frame(right, bg="#1e1e2e")
        self.btn_frame.pack(fill="x", padx=4)
        self._build_class_buttons()

        tk.Frame(right, bg="#45475a", height=1).pack(fill="x", pady=6)

        tk.Button(
            right, text="← Visszavonás  [←]", command=self._go_prev,
            bg="#313244", fg="#cdd6f4", relief="flat", pady=6, width=20,
            font=("Segoe UI", 9),
        ).pack(pady=2, fill="x", padx=4)

        tk.Button(
            right, text="⏭ Kihagyás  [S]", command=self._skip,
            bg="#45475a", fg="#f38ba8", relief="flat", pady=6, width=20,
            font=("Segoe UI", 9),
        ).pack(pady=2, fill="x", padx=4)

        tk.Frame(right, bg="#45475a", height=1).pack(fill="x", pady=6)
        tk.Label(
            right,
            text="1-9 / Q,W,E,R → osztályok\nO → jóváhagy (egyszerűknél)\nS → kihagyás\n← → visszavonás",
            bg="#1e1e2e", fg="#6c7086",
            font=("Consolas", 7), justify="left",
        ).pack(anchor="w", padx=6)

        tk.Frame(right, bg="#45475a", height=1).pack(fill="x", pady=6)
        tk.Label(
            right, text="Session számláló:",
            bg="#1e1e2e", fg="#a6adc8",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=6)
        self.lbl_counter = tk.Label(
            right, text="",
            bg="#1e1e2e", fg="#cdd6f4",
            font=("Consolas", 7), justify="left", wraplength=210,
        )
        self.lbl_counter.pack(anchor="w", padx=6, pady=(2, 4))

    def _build_class_buttons(self) -> None:
        for w in self.btn_frame.winfo_children():
            w.destroy()
        key_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "Q", "W", "E", "R"]
        for i, name in enumerate(CLASS_NAMES):
            key = key_labels[i]
            color = CLASS_COLORS.get(i, "#888888")
            tk.Button(
                self.btn_frame,
                text=f"{key}: {name}",
                command=lambda c=i: self._assign_class(c),
                bg=color, fg="#1e1e2e", relief="flat",
                pady=3, width=22,
                font=("Segoe UI", 8, "bold"),
                activebackground=color,
            ).pack(pady=1, fill="x")

    # ── Billentyűkötések ──────────────────────────────────────────────────────

    def _bind_keys(self) -> None:
        self.root.bind("o", lambda _: self._approve())
        self.root.bind("O", lambda _: self._approve())
        for key, cls_id in KEY_MAP.items():
            self.root.bind(key, lambda _, c=cls_id: self._assign_class(c))
            self.root.bind(key.upper(), lambda _, c=cls_id: self._assign_class(c))
        self.root.bind("s", lambda _: self._skip())
        self.root.bind("S", lambda _: self._skip())
        self.root.bind("<Left>", lambda _: self._go_prev())

    # ── Kép betöltés ──────────────────────────────────────────────────────────

    def _load_current_image(self) -> None:
        if self.img_idx >= self.total:
            self._show_done()
            return

        img_path, folder_name, default_new_id, coco_id = self.images[self.img_idx]
        self._current_coco_id = coco_id
        self._current_default_id = default_new_id
        self._current_boxes = read_txt(img_path.with_suffix(".txt"))

        if default_new_id is not None:
            approve_name = CLASS_NAMES[default_new_id]
            self.btn_approve.config(
                state="normal",
                text=f"✓ {approve_name}  [O]",
                bg="#a6e3a1",
            )
        else:
            self.btn_approve.config(
                state="disabled",
                text="Válassz osztályt ↓",
                bg="#45475a",
            )

        self._refresh_canvas()
        self._refresh_labels()

    def _refresh_canvas(self) -> None:
        img_path = self.images[self.img_idx][0]
        img = render_image(img_path, self._current_boxes, self._current_coco_id)
        img = fit_image(img, MAX_CANVAS_W, MAX_CANVAS_H)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _refresh_labels(self) -> None:
        img_path, folder_name, default_new_id, _ = self.images[self.img_idx]
        self.lbl_progress.config(text=f"{self.img_idx + 1} / {self.total}")
        self.lbl_filename.config(text=img_path.name)
        if default_new_id is not None:
            self.lbl_source.config(
                text=f"forrás: {folder_name}/  →  {CLASS_NAMES[default_new_id]}"
            )
        else:
            self.lbl_source.config(
                text=f"forrás: {folder_name}/  →  ???  (kötelező választás)"
            )

    # ── Osztályozási műveletek ────────────────────────────────────────────────

    def _approve(self) -> None:
        if self.img_idx >= self.total:
            return
        default_id = self.images[self.img_idx][2]
        if default_id is None:
            return
        self._assign_class(default_id)

    def _assign_class(self, new_cls_id: int) -> None:
        if self.img_idx >= self.total:
            return

        img_path, _, _, coco_id = self.images[self.img_idx]
        txt_path = img_path.with_suffix(".txt")
        meta_path = img_path.with_suffix(".json")

        new_boxes = remap_boxes(self._current_boxes, coco_id, new_cls_id)

        cls_name = CLASS_NAMES[new_cls_id]
        target_dir = self.output_dir / cls_name
        target_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        shutil.copy2(img_path, target_dir / img_path.name)
        saved.append(target_dir / img_path.name)
        write_txt(target_dir / txt_path.name, new_boxes)
        saved.append(target_dir / txt_path.name)
        if meta_path.exists():
            shutil.copy2(meta_path, target_dir / meta_path.name)
            saved.append(target_dir / meta_path.name)

        self._history.append(tuple(saved))
        self._session_counts[cls_name] = self._session_counts.get(cls_name, 0) + 1
        self._refresh_counter()
        self._advance()

    def _skip(self) -> None:
        if self.img_idx >= self.total:
            return

        img_path = self.images[self.img_idx][0]
        skip_dir = self.output_dir / "skipped"
        skip_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for ext in (".jpg", ".jpeg", ".png", ".txt", ".json"):
            src = img_path.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, skip_dir / src.name)
                saved.append(skip_dir / src.name)

        self._history.append(tuple(saved))
        self._session_skipped += 1
        self._refresh_counter()
        self._advance()

    def _advance(self) -> None:
        self.img_idx += 1
        self._load_current_image()

    def _go_prev(self) -> None:
        """Visszavon: törli az utolsó kimenetbe mentett fájlokat és visszalép."""
        if self.img_idx == 0 or not self._history:
            return

        last_saved = self._history.pop()
        for f in last_saved:
            if f.exists():
                f.unlink()
            try:
                f.parent.rmdir()
            except OSError:
                pass

        self.img_idx -= 1
        # Visszavonásnál csökkentjük a számlálót
        if last_saved:
            parent_name = last_saved[0].parent.name
            if parent_name == "skipped":
                self._session_skipped = max(0, self._session_skipped - 1)
            elif parent_name in self._session_counts:
                self._session_counts[parent_name] = max(0, self._session_counts[parent_name] - 1)
            self._refresh_counter()
        self._load_current_image()

    def _refresh_counter(self) -> None:
        """Frissíti a session-számláló feliratot – csak a nem-nulla osztályokat mutatja."""
        lines = [
            f"{name}: {count}"
            for name, count in self._session_counts.items()
            if count > 0
        ]
        if self._session_skipped > 0:
            lines.append(f"kihagyva: {self._session_skipped}")
        self.lbl_counter.config(text="\n".join(lines) if lines else "–")

    # ── Befejezés ─────────────────────────────────────────────────────────────

    def _show_done(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            MAX_CANVAS_W // 2, MAX_CANVAS_H // 2,
            text="Kész!\nMinden kép feldolgozva.",
            fill="#a6e3a1",
            font=("Segoe UI", 18, "bold"),
            justify="center",
        )
        self.lbl_progress.config(text=f"Kész – {self.total} kép")
        self.btn_approve.config(state="disabled", text="—", bg="#313244")

        reviewed_counts: dict[str, int] = {}
        if self.output_dir.exists():
            for d in sorted(self.output_dir.iterdir()):
                if d.is_dir() and d.name != "skipped":
                    n = sum(1 for _ in d.glob("*.jpg")) + sum(1 for _ in d.glob("*.png"))
                    if n > 0:
                        reviewed_counts[d.name] = n

        skipped_dir = self.output_dir / "skipped"
        skipped_n = (
            sum(1 for _ in skipped_dir.glob("*.jpg"))
            + sum(1 for _ in skipped_dir.glob("*.png"))
        ) if skipped_dir.exists() else 0

        summary = "\n".join(f"  {k}: {v}" for k, v in reviewed_counts.items())
        if skipped_n:
            summary += f"\n  skipped: {skipped_n}"
        total_reviewed = sum(reviewed_counts.values())

        messagebox.showinfo(
            "Kész",
            f"Összes kép feldolgozva!\n\n"
            f"Kimenet ({self.output_dir}):\n{summary}\n\n"
            f"Összesen: {total_reviewed} jóváhagyva, {skipped_n} kihagyva\n\n"
            f"Következő lépés:\n"
            f"python scripts/prepare_dataset.py --custom_root {self.output_dir}",
        )

    def run(self) -> None:
        if self.total == 0:
            total_in_source = sum(
                sum(1 for _ in (self.source_dir / f).glob("*.jpg"))
                for f in FOLDER_MAP
                if (self.source_dir / f).is_dir()
            )
            if total_in_source == 0:
                print(f"[HIBA] Nincs kép a forrásban: {self.source_dir}")
                print(f"  Előbb futtasd: python scripts/extract_frames_gui.py")
                sys.exit(1)
            else:
                print(f"[OK] Minden kép már felülvizsgálva.")
                print(f"     Új képek bővítéséhez futtasd az extract_frames_gui.py-t,")
                print(f"     majd indítsd újra ezt a programot.")
                self._show_done()
        else:
            print(f"[OK] {self.total} új (még nem felülvizsgált) kép betöltve")
            print(f"     Kimenet: {self.output_dir}")
        self.root.mainloop()


# ── Argumentumok + main ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotáció Átnéző – 13 osztályos manuális felülvizsgáló"
    )
    p.add_argument(
        "--source", type=Path,
        default=Path("C:/Users/admin/Trafic_mojo_2/Training_pictures"),
        help="Forrásmappa (extract_frames.py kimenete, alapért.: Training_pictures)",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("C:/Users/admin/Trafic_mojo_2/Training_pictures_reviewed"),
        help="Kimeneti mappa a felülvizsgált képeknek (alapért.: Training_pictures_reviewed)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        print(f"[HIBA] Forrásmappa nem található: {args.source}")
        print(f"  Előbb futtasd: python scripts/extract_frames_gui.py")
        sys.exit(1)

    print(f"[OK] Forrás:  {args.source}")
    print(f"[OK] Kimenet: {args.output}")
    app = ReviewApp(args.source, args.output)
    app.run()


if __name__ == "__main__":
    main()
