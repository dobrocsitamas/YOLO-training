"""
review_crops.py
---------------
Crop QC eszköz a vehicle classifier tanítóadatához.

Pontosan azt a kivágást mutatja, amit a classifier kapna (+10% margó, 224×224).
Rossz képeket a _rejected almappába helyezi (nem törli!).

Billentyűk:
  Space / → / D   = OK, következő
  X / Delete       = Elutasít → _rejected almappába kerül (.jpg + .txt + .json)
  ← / A            = Vissza (csak megtekintés, elutasítás nem vonható vissza)
  PageDown         = Ugrás +10
  PageUp           = Ugrás -10
  Q / Escape       = Kilépés

Futtatás:
  .venv\Scripts\python scripts\review_crops.py
  .venv\Scripts\python scripts\review_crops.py --cls light_truck
  .venv\Scripts\python scripts\review_crops.py --cls heavy_truck --start 50
"""

import argparse
import shutil
from pathlib import Path
from tkinter import Tk, Canvas, Label, Frame, StringVar, Radiobutton, Button
from tkinter import font as tkfont
from PIL import Image, ImageTk, ImageDraw

# ── Konfiguráció ───────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"C:\Users\admin\Trafic_mojo_2\1_classifier_data")
CLASSES     = [
    "light_truck", "medium_truck", "heavy_truck",
    "vehicle_combination", "bus_solo", "bus_articulated",
]
# Megjegyzés: train_classifier.py medium_truck+heavy_truck → "heavy_truck"-ként tanítja
CROP_SIZE   = 224
MARGIN_FRAC = 0.10   # Ugyanaz mint train_classifier.py-ban!
DISPLAY_SIZE = 448   # 2× zoom a jobb láthatóságért


# ══════════════════════════════════════════════════════════════════════════════
# Crop kinyerés (azonos logika mint train_classifier.py-ban)
# ══════════════════════════════════════════════════════════════════════════════

def get_crop(jpg_path: Path) -> tuple[Image.Image | None, tuple | None]:
    """
    Visszaadja a kivágott képet és a bbox koordinátákat (x1,y1,x2,y2 pixelben).
    Ha nem sikerül, None, None.
    """
    txt_path = jpg_path.with_suffix(".txt")
    if not txt_path.exists():
        return None, None

    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None, None

    best = None
    best_area = 0.0
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = map(float, parts)
        area = w * h
        if area > best_area:
            best_area = area
            best = (cx, cy, w, h)

    if best is None:
        return None, None

    cx, cy, w, h = best
    img = Image.open(jpg_path).convert("RGB")
    iw, ih = img.size

    w_m = w  * (1 + 2 * MARGIN_FRAC)
    h_m = h  * (1 + 2 * MARGIN_FRAC)
    x1 = max(0, int((cx - w_m / 2) * iw))
    y1 = max(0, int((cy - h_m / 2) * ih))
    x2 = min(iw, int((cx + w_m / 2) * iw))
    y2 = min(ih, int((cy + h_m / 2) * ih))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = img.crop((x1, y1, x2, y2)).resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS
    )
    return crop, (x1, y1, x2, y2)


def get_full_thumb(jpg_path: Path, bbox: tuple | None,
                   max_w=640, max_h=360) -> Image.Image:
    """Teljes frame kicsinyítve, bbox piros kerettel jelölve."""
    img = Image.open(jpg_path).convert("RGB")
    iw, ih = img.size

    if bbox:
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    ratio = min(max_w / iw, max_h / ih)
    new_w, new_h = int(iw * ratio), int(ih * ratio)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════

class ReviewApp:
    def __init__(self, root: Tk, cls_name: str, start_idx: int):
        self.root     = root
        self.cls_name = cls_name
        self.folder   = DATA_DIR / cls_name
        self.rejected = DATA_DIR / cls_name / "_rejected"

        # Képlista (csak .jpg, _rejected almappát kizárva)
        self.images = sorted([
            p for p in self.folder.glob("*.jpg")
            if "_rejected" not in str(p)
        ])
        self.idx = max(0, min(start_idx, len(self.images) - 1))

        self._build_ui()
        self._show()

    # ── UI felépítés ───────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title(f"Crop Review – {self.cls_name}")
        self.root.configure(bg="#1e1e1e")
        self.root.bind("<Key>", self._on_key)

        bold16 = tkfont.Font(family="Consolas", size=16, weight="bold")
        norm12 = tkfont.Font(family="Consolas", size=12)
        norm10 = tkfont.Font(family="Consolas", size=10)

        # ── Fejléc ────────────────────────────────────────────────────────────
        hdr = Frame(self.root, bg="#1e1e1e")
        hdr.pack(fill="x", padx=10, pady=(8, 0))

        self.lbl_status = Label(hdr, text="", font=bold16,
                                bg="#1e1e1e", fg="#d4d4d4")
        self.lbl_status.pack(side="left")

        self.lbl_rejected = Label(hdr, text="", font=norm12,
                                  bg="#1e1e1e", fg="#e07070")
        self.lbl_rejected.pack(side="right")

        # ── Képterület ────────────────────────────────────────────────────────
        img_row = Frame(self.root, bg="#1e1e1e")
        img_row.pack(padx=10, pady=6)

        # Bal: crop
        left = Frame(img_row, bg="#1e1e1e")
        left.pack(side="left", padx=(0, 16))
        Label(left, text="CROP (classifier bemenet)",
              font=norm10, bg="#1e1e1e", fg="#888").pack()
        self.canvas_crop = Canvas(left, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                  bg="#2d2d2d", highlightthickness=0)
        self.canvas_crop.pack()

        # Jobb: full frame
        right = Frame(img_row, bg="#1e1e1e")
        right.pack(side="left")
        Label(right, text="TELJES FRAME (piros = bbox)",
              font=norm10, bg="#1e1e1e", fg="#888").pack()
        self.canvas_full = Canvas(right, width=640, height=360,
                                  bg="#2d2d2d", highlightthickness=0)
        self.canvas_full.pack()

        # ── Fájlnév ───────────────────────────────────────────────────────────
        self.lbl_fname = Label(self.root, text="", font=norm10,
                               bg="#1e1e1e", fg="#666", wraplength=1100)
        self.lbl_fname.pack()

        # ── Gombok ────────────────────────────────────────────────────────────
        btn_row = Frame(self.root, bg="#1e1e1e")
        btn_row.pack(pady=10)

        btn_cfg = dict(font=bold16, width=14, relief="flat", cursor="hand2")

        Button(btn_row, text="◀  Vissza  [A/←]",
               bg="#3a3a3a", fg="#d4d4d4",
               command=self._prev, **btn_cfg).pack(side="left", padx=6)

        Button(btn_row, text="✓  OK  [Space/→]",
               bg="#2d6a2d", fg="#ffffff",
               command=self._accept, **btn_cfg).pack(side="left", padx=6)

        Button(btn_row, text="✗  Elutasít  [X/Del]",
               bg="#8b2020", fg="#ffffff",
               command=self._reject, **btn_cfg).pack(side="left", padx=6)

        # ── Osztályváltó ──────────────────────────────────────────────────────
        cls_row = Frame(self.root, bg="#1e1e1e")
        cls_row.pack(pady=(0, 8))
        Label(cls_row, text="Osztály:", font=norm10,
              bg="#1e1e1e", fg="#888").pack(side="left", padx=(0, 6))
        self.cls_var = StringVar(value=self.cls_name)
        for cls in CLASSES:
            rb = Radiobutton(cls_row, text=cls, variable=self.cls_var,
                             value=cls, font=norm10,
                             bg="#1e1e1e", fg="#aaa",
                             selectcolor="#333",
                             activebackground="#1e1e1e",
                             command=self._switch_class)
            rb.pack(side="left", padx=4)

    # ── Megjelenítés ──────────────────────────────────────────────────────────
    def _show(self):
        if not self.images:
            self.lbl_status.config(text="Nincs kép ebben az osztályban.")
            return

        total    = len(self.images)
        rejected = self._count_rejected()

        jpg_path = self.images[self.idx]
        crop, bbox = get_crop(jpg_path)

        # Status
        self.lbl_status.config(
            text=f"[{self.cls_name}]  {self.idx + 1} / {total}"
        )
        self.lbl_rejected.config(
            text=f"Elutasítva: {rejected}"
        )
        self.lbl_fname.config(text=str(jpg_path.name))

        # Crop canvas
        self.canvas_crop.delete("all")
        if crop:
            self._tk_crop = ImageTk.PhotoImage(crop)
            self.canvas_crop.create_image(0, 0, anchor="nw", image=self._tk_crop)
        else:
            self.canvas_crop.create_text(
                DISPLAY_SIZE // 2, DISPLAY_SIZE // 2,
                text="Nincs .txt\nFallback: egész kép",
                fill="#e07070", font=("Consolas", 14), justify="center"
            )

        # Full frame canvas
        self.canvas_full.delete("all")
        thumb = get_full_thumb(jpg_path, bbox)
        self._tk_full = ImageTk.PhotoImage(thumb)
        tw, th = thumb.size
        self.canvas_full.create_image(
            (640 - tw) // 2, (360 - th) // 2,
            anchor="nw", image=self._tk_full
        )

    def _count_rejected(self) -> int:
        if not self.rejected.exists():
            return 0
        return len(list(self.rejected.glob("*.jpg")))

    # ── Navigáció ─────────────────────────────────────────────────────────────
    def _next(self):
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self._show()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _accept(self):
        self._next()

    def _reject(self):
        if not self.images:
            return
        jpg = self.images[self.idx]
        self.rejected.mkdir(exist_ok=True)

        for ext in [".jpg", ".txt", ".json"]:
            src = jpg.with_suffix(ext)
            if src.exists():
                shutil.move(str(src), str(self.rejected / src.name))

        # Frissítjük a listát
        self.images.pop(self.idx)
        if self.idx >= len(self.images):
            self.idx = max(0, len(self.images) - 1)
        self._show()

    def _switch_class(self):
        new_cls = self.cls_var.get()
        if new_cls == self.cls_name:
            return
        self.cls_name = new_cls
        self.folder   = DATA_DIR / new_cls
        self.rejected = DATA_DIR / new_cls / "_rejected"
        self.images   = sorted([
            p for p in self.folder.glob("*.jpg")
            if "_rejected" not in str(p)
        ])
        self.idx = 0
        self.root.title(f"Crop Review – {self.cls_name}")
        self._show()

    # ── Billentyűk ────────────────────────────────────────────────────────────
    def _on_key(self, event):
        k = event.keysym.lower()
        if k in ("space", "right", "d"):
            self._accept()
        elif k in ("x", "delete"):
            self._reject()
        elif k in ("left", "a"):
            self._prev()
        elif k == "next":       # PageDown
            self.idx = min(len(self.images) - 1, self.idx + 10)
            self._show()
        elif k == "prior":      # PageUp
            self.idx = max(0, self.idx - 10)
            self._show()
        elif k in ("q", "escape"):
            self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Belépési pont
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls",   default=CLASSES[0],
                        choices=CLASSES, help="Melyik osztállyal kezdje")
    parser.add_argument("--start", type=int, default=0,
                        help="Hányadik képtől induljon")
    args = parser.parse_args()

    root = Tk()
    root.resizable(False, False)
    app = ReviewApp(root, args.cls, args.start)
    root.mainloop()

    # Összefoglaló a kilépés után
    print("\n=== Review összefoglaló ===")
    for cls in CLASSES:
        folder   = DATA_DIR / cls
        rejected = folder / "_rejected"
        n_ok  = len(list(folder.glob("*.jpg")))
        n_rej = len(list(rejected.glob("*.jpg"))) if rejected.exists() else 0
        if n_ok + n_rej > 0:
            print(f"  {cls:<22}: {n_ok:>4} OK  |  {n_rej:>3} elutasítva")


if __name__ == "__main__":
    main()
