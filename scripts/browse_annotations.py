#!/usr/bin/env python3
"""
Annotáció böngésző – bounding boxok és osztálycímkék megjelenítése
==================================================================
Használat:
  python scripts/browse_annotations.py
  python scripts/browse_annotations.py --folder personal_car
  python scripts/browse_annotations.py --root C:\...\Arhív\Training_pictures_reviewed

Billentyűk:
  → / D    : következő kép
  ← / A    : előző kép
  PageDown : +10 kép
  PageUp   : -10 kép
  F        : mappát vált (folder selector)
  Q / Esc  : kilépés
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    print("[HIBA] Pillow vagy tkinter hiányzik. Futtasd: pip install Pillow")
    sys.exit(1)

# ── osztályok ────────────────────────────────────────────────────────────────

CLASS_NAMES_TRAFFIC14 = [
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
    "minibus",             # 13
]

# COCO osztályok (csak a releváns ID-k)
CLASS_NAMES_COCO = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    4:  "airplane",
    5:  "bus",
    6:  "train",
    7:  "truck",
    8:  "boat",
    14: "bird",
    15: "cat",
    16: "dog",
}

# Aktív schema – program indításkor Traffic14
CLASS_NAMES = CLASS_NAMES_TRAFFIC14

CLASS_COLORS = [
    "#a6e3a1",  # 0  person        – zöld
    "#89dceb",  # 1  bicycle       – cyan
    "#89b4fa",  # 2  motorcycle    – kék
    "#b4befe",  # 3  personal_car  – lila
    "#a6e3a1",  # 4  light_truck   – zöld
    "#fab387",  # 5  medium_truck  – narancs
    "#f38ba8",  # 6  heavy_truck   – piros
    "#cba6f7",  # 7  veh_combo     – lila
    "#fe640b",  # 8  bus_solo      – narancs
    "#e64553",  # 9  bus_artic     – piros
    "#04a5e5",  # 10 trolley_solo  – kék
    "#209fb5",  # 11 trolley_artic – teal
    "#7287fd",  # 12 tram          – indigo
    "#f9e2af",  # 13 minibus       – sárga
]

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

MAX_W = 1100
MAX_H = 680

LABEL_FONT_SIZE = 16   # bbox felirat betűmérete

KNOWN_ROOTS = {
    "Jelenlegi (Training_pictures_reviewed)":
        Path(r"C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed"),
    "Archív (Arhív\\Training_pictures_reviewed)":
        Path(r"C:\Users\admin\Trafic_mojo_2\Arhív\Training_pictures_reviewed"),
    "Eredeti gyűjtött (Training_pictures)":
        Path(r"C:\Users\admin\Trafic_mojo_2\Training_pictures"),
    "Munkaközi / training_data_DONT_USE":
        Path(r"C:\Users\admin\Trafic_mojo_2\Munkaközi\training_data_DONT_USE"),
}

DEFAULT_ROOT = Path(r"C:\Users\admin\Trafic_mojo_2\Training_pictures_reviewed")
ARCHIVE_ROOT = Path(r"C:\Users\admin\Trafic_mojo_2\Arhív\Training_pictures_reviewed")


def read_txt(txt_path: Path):
    boxes = []
    if txt_path.exists():
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    boxes.append((int(parts[0]), float(parts[1]),
                                  float(parts[2]), float(parts[3]), float(parts[4])))
                except ValueError:
                    pass
    return boxes


def get_class_name(cls_id: int, schema: str) -> str:
    if schema == "COCO":
        return CLASS_NAMES_COCO.get(cls_id, f"cls{cls_id}")
    else:
        if cls_id < len(CLASS_NAMES_TRAFFIC14):
            return CLASS_NAMES_TRAFFIC14[cls_id]
        return f"cls{cls_id}"


def render(img_path: Path, boxes, folder_name: str, schema: str = "Traffic14") -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # Melyik osztály az "elvárt" trigger ebben a mappában?
    trigger_id = None
    for i, name in enumerate(CLASS_NAMES):
        if name == folder_name:
            trigger_id = i
            break

    # Font betöltés – nagyobb méret
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", LABEL_FONT_SIZE)
    except Exception:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", LABEL_FONT_SIZE)
        except Exception:
            font = ImageFont.load_default()

    for cls, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        color_hex = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else "#ffffff"
        color_rgb = hex_to_rgb(color_hex)

        is_trigger = (cls == trigger_id)
        lw = 5 if is_trigger else 2

        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=lw)

        label = get_class_name(cls, schema)
        if is_trigger:
            label = f"★ {label}"

        # Szöveg méretének meghatározása
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0] + 6
            th = bbox[3] - bbox[1] + 4
        except Exception:
            tw = len(label) * 9 + 6
            th = LABEL_FONT_SIZE + 4

        ty = max(y1 - th - 2, 0)
        draw.rectangle([x1, ty, x1 + tw, ty + th], fill=color_rgb)
        draw.text((x1 + 3, ty + 2), label, fill=(0, 0, 0), font=font)

    # Átméretezés ha kell
    scale = min(MAX_W / W, MAX_H / H, 1.0)
    if scale < 1.0:
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    return img


class BrowseApp:
    def __init__(self, root_path: Path, start_folder: str = None):
        self.root_path = root_path

        # ── Tkinter ablak ────────────────────────────────────────────────
        self.win = tk.Tk()
        self.win.title("Annotáció böngésző")
        self.win.configure(bg="#1e1e2e")

        # ── Root (forrásmappa) választó ──────────────────────────────────
        root_bar = tk.Frame(self.win, bg="#181825")
        root_bar.pack(fill="x", padx=4, pady=(4, 0))

        tk.Label(root_bar, text="Forrásmappa:", bg="#181825", fg="#f9e2af",
                 font=("Consolas", 9)).pack(side="left", padx=4)

        self._root_names = list(KNOWN_ROOTS.keys())
        # Megtaláljuk a kezdő root nevét
        default_root_name = self._root_names[0]
        for k, v in KNOWN_ROOTS.items():
            if v == root_path:
                default_root_name = k
                break
        self.root_var = tk.StringVar(value=default_root_name)
        self.root_cb = ttk.Combobox(root_bar, textvariable=self.root_var,
                               values=self._root_names, width=55, state="readonly")
        self.root_cb.pack(side="left", padx=4)
        self.root_cb.bind("<<ComboboxSelected>>", self._on_root_change)

        tk.Button(root_bar, text="📁 Tallózás...", command=self._browse_root,
                  bg="#45475a", fg="#cdd6f4", relief="flat", padx=8).pack(side="left", padx=4)

        # ── Osztálymappa toolbar ─────────────────────────────────────────
        tb = tk.Frame(self.win, bg="#313244")
        tb.pack(fill="x", padx=4, pady=4)

        tk.Label(tb, text="Osztály:", bg="#313244", fg="#cdd6f4").pack(side="left", padx=4)
        self.folder_var = tk.StringVar()
        self.folder_cb = ttk.Combobox(tb, textvariable=self.folder_var,
                                      width=24, state="readonly")
        self.folder_cb.pack(side="left")
        self.folder_cb.bind("<<ComboboxSelected>>", self._on_folder_change)

        self.info_label = tk.Label(tb, text="", bg="#313244", fg="#a6e3a1",
                                   font=("Consolas", 10))
        self.info_label.pack(side="left", padx=16)

        # Schema váltó
        tk.Label(tb, text="Kategória-séma:", bg="#313244", fg="#f9e2af",
                 font=("Consolas", 9)).pack(side="right", padx=(0, 4))
        self.schema_var = tk.StringVar(value="Traffic14")
        for schema in ("Traffic14", "COCO"):
            tk.Radiobutton(tb, text=schema, variable=self.schema_var, value=schema,
                           bg="#313244", fg="#cdd6f4", selectcolor="#45475a",
                           activebackground="#313244", activeforeground="#cdd6f4",
                           command=self._show).pack(side="right", padx=2)

        # Navigáció
        nav = tk.Frame(self.win, bg="#313244")
        nav.pack(fill="x", padx=4)

        btn = lambda t, cmd: tk.Button(nav, text=t, command=cmd,
                                       bg="#45475a", fg="#cdd6f4",
                                       relief="flat", padx=8)
        btn("◀◀ -10", lambda: self._jump(-10)).pack(side="left", padx=2)
        btn("◀ Előző", lambda: self._jump(-1)).pack(side="left", padx=2)
        btn("Következő ▶", lambda: self._jump(1)).pack(side="left", padx=2)
        btn("+10 ▶▶", lambda: self._jump(10)).pack(side="left", padx=2)
        btn("📂 Archív összehasonlít.", self._show_archive_diff).pack(side="left", padx=16)

        # Kép canvas
        self.canvas = tk.Label(self.win, bg="#1e1e2e")
        self.canvas.pack(padx=4, pady=4)

        # Annotáció lista
        self.ann_label = tk.Label(self.win, text="", bg="#1e1e2e", fg="#cdd6f4",
                                  font=("Consolas", 10), justify="left", anchor="w")
        self.ann_label.pack(fill="x", padx=8, pady=(0, 4))

        # Billentyűk
        self.win.bind("<Right>",    lambda e: self._jump(1))
        self.win.bind("<Left>",     lambda e: self._jump(-1))
        self.win.bind("<d>",        lambda e: self._jump(1))
        self.win.bind("<a>",        lambda e: self._jump(-1))
        self.win.bind("<Next>",     lambda e: self._jump(10))
        self.win.bind("<Prior>",    lambda e: self._jump(-10))
        self.win.bind("<q>",        lambda e: self.win.quit())
        self.win.bind("<Escape>",   lambda e: self.win.quit())

        # Mappalista betöltése és megjelenítés
        self._reload_folders(start_folder=start_folder)
        self.win.mainloop()

    def _reload_folders(self, start_folder: str = None):
        """Root-váltáskor vagy inicializáláskor újratölti a mappalistát."""
        self.folders = sorted([
            d.name for d in self.root_path.iterdir()
            if d.is_dir() and d.name != "skipped"
        ])
        if not self.folders:
            self.folders = ["(üres)"]
        self.folder_cb.config(values=self.folders)
        self.folder_idx = 0
        if start_folder and start_folder in self.folders:
            self.folder_idx = self.folders.index(start_folder)
        self.folder_var.set(self.folders[self.folder_idx])
        self.img_idx = 0
        self.images = []
        self._load_folder()
        self._show()

    def _on_root_change(self, _event=None):
        chosen = self.root_var.get()
        new_root = KNOWN_ROOTS.get(chosen, DEFAULT_ROOT)
        if not new_root.is_dir():
            self.info_label.config(text=f"⚠ Nem létezik: {new_root}")
            return
        self.root_path = new_root
        self._reload_folders()

    def _browse_root(self):
        """Tetszőleges mappa kiválasztása fájlböngészővel."""
        chosen = filedialog.askdirectory(
            title="Válaszd ki a képmappát",
            initialdir=str(self.root_path),
        )
        if not chosen:
            return
        new_root = Path(chosen)
        if not new_root.is_dir():
            return
        self.root_path = new_root
        # Ha a mappának nincs almappája, direktben tartalmaz képeket →
        # próbáljuk egy szinttel feljebb is megmutatni
        subdirs = [d for d in new_root.iterdir() if d.is_dir() and d.name != "skipped"]
        if not subdirs:
            # Lehet hogy maga a mappa tartalmazza a képeket (nem almappa-struktúra)
            # Ilyenkor a szülőmappát Root-nak, az aktuálist almappának tekintjük
            self.root_path = new_root.parent
        # Frissítjük a combobox értékét (egyedi elérési út megjelenítése)
        display = str(new_root)
        if display not in self._root_names:
            self._root_names.append(display)
            self.root_cb.config(values=self._root_names)
        self.root_var.set(display)
        self._reload_folders(start_folder=new_root.name if not subdirs else None)

    def _load_folder(self):
        if not self.folders or self.folders[0] == "(üres)":
            self.images = []
            return
        folder_path = self.root_path / self.folders[self.folder_idx]
        self.images = sorted([
            p for p in folder_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        self.img_idx = 0

    def _on_folder_change(self, _event=None):
        self.folder_idx = self.folders.index(self.folder_var.get())
        self._load_folder()
        self._show()

    def _jump(self, delta: int):
        if not self.images:
            return
        self.img_idx = max(0, min(len(self.images) - 1, self.img_idx + delta))
        self._show()

    def _show(self):
        if not self.images:
            self.info_label.config(text="(Nincs kép)")
            self.canvas.config(image="")
            self.ann_label.config(text="")
            return

        img_path = self.images[self.img_idx]
        txt_path = img_path.with_suffix(".txt")
        boxes = read_txt(txt_path)
        folder_name = self.folders[self.folder_idx]
        schema = self.schema_var.get()

        pil_img = render(img_path, boxes, folder_name, schema)
        self._tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.config(image=self._tk_img)

        self.info_label.config(
            text=f"{folder_name}  |  {self.img_idx+1}/{len(self.images)}  |  {img_path.name}"
        )

        if boxes:
            lines = []
            for cls, cx, cy, bw, bh in boxes:
                name = get_class_name(cls, schema)
                trigger_id = next((i for i, n in enumerate(CLASS_NAMES_TRAFFIC14)
                                   if n == folder_name), None)
                trigger_mark = " ★" if cls == trigger_id else ""
                lines.append(f"  cls{cls}  {name}{trigger_mark}   cx={cx:.3f} cy={cy:.3f} w={bw:.3f} h={bh:.3f}")
            self.ann_label.config(text="\n".join(lines))
        else:
            self.ann_label.config(text="  (nincs annotáció – üres .txt)")

    def _show_archive_diff(self):
        if not self.images:
            return
        img_path = self.images[self.img_idx]
        folder_name = self.folders[self.folder_idx]
        archive_txt = ARCHIVE_ROOT / folder_name / img_path.with_suffix(".txt").name
        current_txt = img_path.with_suffix(".txt")

        win2 = tk.Toplevel(self.win)
        win2.title(f"Összehasonlítás – {img_path.name}")
        win2.configure(bg="#1e1e2e")

        def fmt(p):
            boxes = read_txt(p)
            if not boxes:
                return "(üres / nem létezik)"
            lines = []
            for cls, cx, cy, bw, bh in boxes:
                name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}"
                lines.append(f"  cls{cls}  {name}   cx={cx:.3f} cy={cy:.3f}")
            return "\n".join(lines)

        for title, path in [("JELENLEGI", current_txt), ("ARCHÍV", archive_txt)]:
            tk.Label(win2, text=title, bg="#313244", fg="#f9e2af",
                     font=("Consolas", 11, "bold")).pack(fill="x", padx=8, pady=(8, 0))
            tk.Label(win2, text=str(path), bg="#181825", fg="#6c7086",
                     font=("Consolas", 8)).pack(fill="x", padx=8)
            tk.Label(win2, text=fmt(path), bg="#1e1e2e", fg="#cdd6f4",
                     font=("Consolas", 10), justify="left", anchor="w").pack(
                fill="x", padx=8, pady=(0, 8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="Training_pictures_reviewed mappa")
    parser.add_argument("--folder", default=None,
                        help="Kezdő almappa neve (pl. personal_car)")
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.is_dir():
        print(f"Nem található: {root_path}")
        sys.exit(1)

    BrowseApp(root_path, start_folder=args.folder)


if __name__ == "__main__":
    main()
