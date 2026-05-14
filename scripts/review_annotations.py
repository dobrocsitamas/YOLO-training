#!/usr/bin/env python3
"""
Annotation Review Tool
======================
Manuális annotáció-ellenőrző a kivont képekhez.

Megmutatja az extract_frames.py által kimentett képeket,
és lehetővé teszi a bus/truck alkategória manuális meghatározását.

Működés:
  - Betölti a training_data/raw/bus/ és training_data/raw/truck/ képeit
  - Minden képhez megmutatja az annotált bbox-okat
  - A placeholder bbox-okat (bus=4, truck=7) kiemelve jeleníti meg
  - Egyszerre egy placeholder osztályozható (ha több van a képen: sorban)
  - Osztályozás után a kép + .txt a megfelelő almappába kerül
  - A .txt-ben az adott placeholder class ID-t a kiválasztottra cseréli

Billentyűzet:
  1 → bus_solo         (4)
  2 → bus_articulated  (5)
  3 → truck_light      (6)
  4 → truck_heavy      (7)
  5 → vehicle_combination (8)
  S → kihagyás (képet áthelyezi a 'skipped/' mappába)
  ← → előző kép

Kimenet:
  training_data/reviewed/
    bus_solo/
    bus_articulated/
    truck_light/
    truck_heavy/
    vehicle_combination/
    skipped/

Következő lépés:
  python scripts/prepare_dataset.py --custom_root training_data/reviewed --output dataset

Használat:
  python scripts/review_annotations.py
  python scripts/review_annotations.py --source training_data/raw --output training_data/reviewed
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    print("[HIBA] Pillow nem telepítve. Futtasd: pip install Pillow")
    sys.exit(1)


# ── Osztályok ─────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "person",               # 0
    "bicycle",              # 1
    "car",                  # 2
    "motorcycle",           # 3
    "bus_solo",             # 4
    "bus_articulated",      # 5
    "truck_light",          # 6
    "truck_heavy",          # 7
    "vehicle_combination",  # 8
]

# Osztályozható alkategóriák: {new_class_id: (gomb felirat, hex szín)}
ASSIGNABLE_CLASSES: dict[int, tuple[str, str]] = {
    4: ("1: bus_solo",            "#fe640b"),
    5: ("2: bus_articulated",     "#e64553"),
    6: ("3: truck_light",         "#40a02b"),
    7: ("4: truck_heavy",         "#df8e1d"),
    8: ("5: vehicle_combination", "#8839ef"),
}

# Placeholder ID-k (ezeket kell osztályozni)
BUS_PLACEHOLDER_ID   = 4
TRUCK_PLACEHOLDER_ID = 7
PLACEHOLDER_IDS = {BUS_PLACEHOLDER_ID, TRUCK_PLACEHOLDER_ID}

# Bbox színek (osztályonként)
CLASS_COLORS = {
    0: "#00ff00",  # person
    1: "#00ffff",  # bicycle
    2: "#ffff00",  # car
    3: "#ff00ff",  # motorcycle
    4: "#ff8800",  # bus_solo / placeholder
    5: "#ff4444",  # bus_articulated
    6: "#88ff00",  # truck_light
    7: "#ff2222",  # truck_heavy / placeholder
    8: "#cc44ff",  # vehicle_combination
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


# ── Kép renderelés ────────────────────────────────────────────────────────────

def render_image(
    img_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
    active_box_idx: int | None,
) -> Image.Image:
    """Visszaad egy PIL képet a bbox-okkal. Az active_box vastag/kiemelve."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    for i, (cls, cx, cy, bw, bh) in enumerate(boxes):
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        color = CLASS_COLORS.get(cls, "#ffffff")
        is_active = (i == active_box_idx)
        is_placeholder = cls in PLACEHOLDER_IDS

        lw = 4 if is_active else (2 if is_placeholder else 1)

        # Dimming: nem aktív és nem placeholder → halványabb
        if not is_active and not is_placeholder:
            color = color[0] + "88"  # alfa-szerű hatás a szöveghez

        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}"
        if is_active:
            label = f"▶ {label} ◀"

        # Szöveg háttér
        try:
            draw.text((x1 + 3, y1 + 2), label, fill=color)
        except Exception:
            pass

    return img


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


# ── Fő alkalmazás ─────────────────────────────────────────────────────────────

class ReviewApp:
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir

        # Képek összegyűjtése bus/ és truck/ almappákból
        self.images: list[Path] = []
        for subfolder in sorted(source_dir.iterdir()):
            if subfolder.is_dir() and subfolder.name in ("bus", "truck"):
                self.images.extend(sorted(subfolder.glob("*.jpg")))

        if not self.images:
            # Fallback: összes .jpg
            self.images = sorted(source_dir.rglob("*.jpg"))

        self.total = len(self.images)
        self.img_idx = 0            # aktuális kép indexe
        self.placeholder_idx = 0    # az aktuális képen belül melyik placeholder sorra kerül

        # Aktuális kép placeholder-jei
        self._current_boxes: list[tuple[int, float, float, float, float]] = []
        self._placeholder_indices: list[int] = []  # box-lista indexei ahol placeholder van

        self.root = tk.Tk()
        self.root.title("Annotáció Átnéző – YOLO Training")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        self._build_ui()
        self._bind_keys()
        self._load_current_image()

    # ── UI felépítés ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Felső sáv
        top = tk.Frame(self.root, bg="#313244", pady=5)
        top.pack(fill="x")
        self.lbl_progress = tk.Label(
            top, text="", bg="#313244", fg="#cdd6f4", font=("Segoe UI", 10)
        )
        self.lbl_progress.pack(side="left", padx=12)
        self.lbl_filename = tk.Label(
            top, text="", bg="#313244", fg="#a6e3a1", font=("Segoe UI", 9)
        )
        self.lbl_filename.pack(side="right", padx=12)

        # Fő terület
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # Canvas (kép)
        self.canvas = tk.Canvas(
            main, bg="#181825",
            width=MAX_CANVAS_W, height=MAX_CANVAS_H,
            highlightthickness=0,
        )
        self.canvas.pack(side="left", padx=(0, 8))

        # Jobb panel
        right = tk.Frame(main, bg="#1e1e2e", width=230)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(
            right, text="Osztályozás", bg="#1e1e2e", fg="#89b4fa",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=(4, 4))

        tk.Label(right, text="Osztályozandó:", bg="#1e1e2e",
                 fg="#a6adc8", font=("Segoe UI", 9)).pack()

        self.lbl_current = tk.Label(
            right, text="—", bg="#1e1e2e", fg="#fab387",
            font=("Segoe UI", 12, "bold")
        )
        self.lbl_current.pack(pady=(2, 8))

        # Alkategória gombok
        self.btn_frame = tk.Frame(right, bg="#1e1e2e")
        self.btn_frame.pack(fill="x", padx=4)
        self._build_class_buttons()

        # Elválasztó
        tk.Frame(right, bg="#45475a", height=1).pack(fill="x", pady=10)

        # Navigáció
        tk.Button(
            right, text="← Előző  [←]", command=self._go_prev,
            bg="#313244", fg="#cdd6f4", relief="flat", pady=6, width=20,
            font=("Segoe UI", 9),
        ).pack(pady=2, fill="x", padx=4)

        tk.Button(
            right, text="⏭ Kihagyás  [S]", command=self._skip,
            bg="#45475a", fg="#f38ba8", relief="flat", pady=6, width=20,
            font=("Segoe UI", 9),
        ).pack(pady=2, fill="x", padx=4)

        # Súgó
        tk.Frame(right, bg="#45475a", height=1).pack(fill="x", pady=10)
        tk.Label(
            right,
            text=(
                "Billentyűzet:\n"
                "  1 → bus_solo\n"
                "  2 → bus_articulated\n"
                "  3 → truck_light\n"
                "  4 → truck_heavy\n"
                "  5 → vehicle_combination\n"
                "  S → kihagyás\n"
                "  ← → előző kép\n\n"
                "Sárga/narancssárga keret\n"
                "= osztályozandó objektum"
            ),
            bg="#1e1e2e", fg="#6c7086",
            font=("Consolas", 8), justify="left",
        ).pack(anchor="w", padx=6)

    def _build_class_buttons(self):
        for w in self.btn_frame.winfo_children():
            w.destroy()
        for cls_id, (label, color) in ASSIGNABLE_CLASSES.items():
            tk.Button(
                self.btn_frame, text=label,
                command=lambda c=cls_id: self._assign_class(c),
                bg=color, fg="white", relief="flat",
                pady=7, width=22,
                font=("Segoe UI", 9, "bold"),
                activebackground=color, activeforeground="white",
            ).pack(pady=2, fill="x")

    # ── Billentyűkötések ──────────────────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("1", lambda _: self._assign_class(4))
        self.root.bind("2", lambda _: self._assign_class(5))
        self.root.bind("3", lambda _: self._assign_class(6))
        self.root.bind("4", lambda _: self._assign_class(7))
        self.root.bind("5", lambda _: self._assign_class(8))
        self.root.bind("s", lambda _: self._skip())
        self.root.bind("S", lambda _: self._skip())
        self.root.bind("<Left>",  lambda _: self._go_prev())

    # ── Kép betöltés / megjelenítés ───────────────────────────────────────────

    def _load_current_image(self):
        if self.img_idx >= self.total:
            self._show_done()
            return

        img_path = self.images[self.img_idx]
        txt_path = img_path.with_suffix(".txt")

        self._current_boxes = read_txt(txt_path)
        self._placeholder_indices = [
            i for i, (cls, *_) in enumerate(self._current_boxes)
            if cls in PLACEHOLDER_IDS
        ]
        self.placeholder_idx = 0

        self._refresh_canvas()
        self._refresh_labels()

    def _refresh_canvas(self):
        img_path = self.images[self.img_idx]
        active_box = (
            self._placeholder_indices[self.placeholder_idx]
            if self.placeholder_idx < len(self._placeholder_indices)
            else None
        )

        img = render_image(img_path, self._current_boxes, active_box)
        img = fit_image(img, MAX_CANVAS_W, MAX_CANVAS_H)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _refresh_labels(self):
        img_path = self.images[self.img_idx]

        # Progress: kép + placeholder
        total_ph = len(self._placeholder_indices)
        ph_info = (
            f"  |  obj: {self.placeholder_idx + 1}/{total_ph}"
            if total_ph > 1 else ""
        )
        self.lbl_progress.config(
            text=f"{self.img_idx + 1} / {self.total}{ph_info}"
        )
        self.lbl_filename.config(text=img_path.name)

        # Jelenlegi osztályozandó
        if self.placeholder_idx < len(self._placeholder_indices):
            box_i = self._placeholder_indices[self.placeholder_idx]
            cls = self._current_boxes[box_i][0]
            self.lbl_current.config(
                text=CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
            )
        else:
            self.lbl_current.config(text="—")

    # ── Osztályozás ───────────────────────────────────────────────────────────

    def _assign_class(self, new_cls_id: int):
        if self.img_idx >= self.total:
            return
        if self.placeholder_idx >= len(self._placeholder_indices):
            return

        # Melyik box-t módosítjuk
        box_i = self._placeholder_indices[self.placeholder_idx]
        old_cls = self._current_boxes[box_i][0]

        # Box frissítése
        cls, cx, cy, w, h = self._current_boxes[box_i]
        self._current_boxes[box_i] = (new_cls_id, cx, cy, w, h)

        # Van még placeholder ezen a képen?
        self.placeholder_idx += 1

        if self.placeholder_idx < len(self._placeholder_indices):
            # Még van placeholder ugyanebben a képben → megjelenítés frissítése
            self._refresh_canvas()
            self._refresh_labels()
        else:
            # Minden placeholder osztályozva → mentés + következő kép
            self._save_current_and_advance(new_cls_id)

    def _save_current_and_advance(self, last_cls_id: int):
        """Elmenti a képet + .txt-t a megfelelő osztálymappába, majd következő."""
        img_path = self.images[self.img_idx]
        txt_path = img_path.with_suffix(".txt")
        meta_path = img_path.with_suffix(".json")

        # A kimeneti mappa neve az utoljára hozzárendelt osztályból
        # (ha több placeholder volt, az utolsóval mentünk – a trigger obj. a leglényegesebb)
        # Az összes placeholder-hez rendelt osztály alapján a "fő" trigger mappát keressük:
        assigned_clses = set()
        for i in self._placeholder_indices:
            assigned_clses.add(self._current_boxes[i][0])

        # Ha több különböző osztály lett rendelve, az utolsót (trigger) vesszük alapnak
        target_cls_name = CLASS_NAMES[last_cls_id] if last_cls_id < len(CLASS_NAMES) else f"class_{last_cls_id}"
        target_dir = self.output_dir / target_cls_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Fájlok másolása
        shutil.copy2(img_path, target_dir / img_path.name)
        write_txt(target_dir / txt_path.name, self._current_boxes)
        if meta_path.exists():
            shutil.copy2(meta_path, target_dir / meta_path.name)

        self._next_image()

    def _skip(self):
        """Kép áthelyezése a skipped/ mappába."""
        if self.img_idx >= self.total:
            return

        img_path = self.images[self.img_idx]
        skip_dir = self.output_dir / "skipped"
        skip_dir.mkdir(parents=True, exist_ok=True)

        for ext in (".jpg", ".txt", ".json"):
            src = img_path.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, skip_dir / src.name)

        self._next_image()

    def _next_image(self):
        self.img_idx += 1
        self.placeholder_idx = 0
        self._load_current_image()

    def _go_prev(self):
        if self.img_idx > 0:
            self.img_idx -= 1
            self.placeholder_idx = 0
            self._load_current_image()

    # ── Befejezés ─────────────────────────────────────────────────────────────

    def _show_done(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            MAX_CANVAS_W // 2, MAX_CANVAS_H // 2,
            text="Kész!\nMinden kép feldolgozva.",
            fill="#a6e3a1",
            font=("Segoe UI", 18, "bold"),
            justify="center",
        )
        self.lbl_progress.config(text=f"Kész – {self.total} kép")
        self.lbl_current.config(text="—")

        reviewed_counts = {}
        if self.output_dir.exists():
            for d in self.output_dir.iterdir():
                if d.is_dir():
                    n = len(list(d.glob("*.jpg")))
                    if n > 0:
                        reviewed_counts[d.name] = n

        summary = "\n".join(f"  {k}: {v} kép" for k, v in sorted(reviewed_counts.items()))
        messagebox.showinfo(
            "Kész",
            f"Összes kép feldolgozva!\n\nKimenet ({self.output_dir}):\n{summary}\n\n"
            f"Következő lépés:\n"
            f"python scripts/prepare_dataset.py \\\n"
            f"  --custom_root {self.output_dir}",
        )

    def run(self):
        if self.total == 0:
            print("[HIBA] Nincs betölthető kép.")
            print(f"  Ellenőrizd, hogy a {self.source_dir}/bus/ és /truck/ mappák tartalmazzák a képeket.")
            sys.exit(1)
        print(f"[OK] {self.total} kép betöltve → átnézés indul")
        self.root.mainloop()


# ── Argumentumok + main ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotáció Átnéző – manuális bus/truck alkategória osztályozó"
    )
    p.add_argument(
        "--source", type=Path, default=Path("training_data/raw"),
        help="Forrásmappa (extract_frames.py kimenete, alapért.: training_data/raw)",
    )
    p.add_argument(
        "--output", type=Path, default=Path("training_data/reviewed"),
        help="Kimeneti mappa (alapért.: training_data/reviewed)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        print(f"[HIBA] Forrásmappa nem található: {args.source}")
        print(f"  Előbb futtasd: python scripts/extract_frames.py --video <videó> --model <modell>")
        sys.exit(1)

    print(f"[OK] Képek betöltése: {args.source}")
    app = ReviewApp(args.source, args.output)
    app.run()


if __name__ == "__main__":
    main()
