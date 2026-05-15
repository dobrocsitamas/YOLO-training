#!/usr/bin/env python3
"""
Frame Extractor – GUI indító
============================
Grafikus felület az extract_frames.py futtatásához.
Indítás:
    python scripts/extract_frames_gui.py
"""

import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, scrolledtext, ttk

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b[\[\]()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><~]")

import cv2

# ── Alapértelmezett útvonalak ─────────────────────────────────────────────────
DEFAULT_MODEL  = r"C:\Users\admin\Trafic_mojo_2\TM_modulok_py\Traffic_Mojo_2_0\yolo11s.pt"
DEFAULT_VIDEO  = r"C:\Users\admin\Trafic_mojo_2\Video"
DEFAULT_OUTPUT = r"C:\Users\admin\Trafic_mojo_2\YOLO-training\training_data\raw"

DEFAULT_TRIGGERS = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

ALL_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mts", ".m2ts", ".wmv", ".ts"}

CHECK_ON  = "☑"
CHECK_OFF = "☐"


# ── Videó időtartam lekérése ──────────────────────────────────────────────────

def get_video_duration(path: Path) -> str:
    try:
        cap = cv2.VideoCapture(str(path))
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0 and frames > 0:
            secs = int(frames / fps)
            h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
            return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        pass
    return "—"


def scan_videos(path_str: str) -> list[Path]:
    p = Path(path_str)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
    return []


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Frame Extractor – képgyűjtő")
        self.resizable(True, True)
        self.minsize(780, 700)

        self._process: subprocess.Popen | None = None
        self._running = False

        self._build_ui()
        self._center_window()
        self.after(100, self._scan_videos)

    # ── UI felépítés ──────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # ── Modell és kimenet ─────────────────────────────────────────────────
        paths_frame = ttk.LabelFrame(self, text="Modell és kimenet", padding=8)
        paths_frame.pack(fill="x", **pad)
        paths_frame.columnconfigure(1, weight=1)

        self._model_var  = tk.StringVar(value=DEFAULT_MODEL)
        self._output_var = tk.StringVar(value=DEFAULT_OUTPUT)

        self._add_path_row(paths_frame, 0, "Modell (.pt):",
                           self._model_var, self._browse_model)
        self._add_path_row(paths_frame, 1, "Kimeneti mappa:",
                           self._output_var, self._browse_output)

        # ── Videók ────────────────────────────────────────────────────────────
        vid_frame = ttk.LabelFrame(self, text="Videók", padding=8)
        vid_frame.pack(fill="both", expand=False, **pad)
        vid_frame.columnconfigure(1, weight=1)

        self._video_var = tk.StringVar(value=DEFAULT_VIDEO)
        self._video_var.trace_add("write", lambda *_: self.after(300, self._scan_videos))

        ttk.Label(vid_frame, text="Mappa / fájl:").grid(
            row=0, column=0, sticky="e", padx=(0, 6), pady=3)
        ttk.Entry(vid_frame, textvariable=self._video_var, width=60).grid(
            row=0, column=1, sticky="ew", pady=3)

        btn_inner = ttk.Frame(vid_frame)
        btn_inner.grid(row=0, column=2, padx=(6, 0))
        ttk.Button(btn_inner, text="Fájl…",  command=self._browse_video_file,
                   width=7).pack(side="left", padx=(0, 4))
        ttk.Button(btn_inner, text="Mappa…", command=self._browse_video_dir,
                   width=7).pack(side="left")

        tree_frame = ttk.Frame(vid_frame)
        tree_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        vid_frame.rowconfigure(1, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("check", "name", "duration", "size"),
            show="headings",
            height=6,
            selectmode="none",
        )
        self._tree.heading("check",    text="")
        self._tree.heading("name",     text="Fájlnév")
        self._tree.heading("duration", text="Hossz")
        self._tree.heading("size",     text="Méret")

        self._tree.column("check",    width=30,  stretch=False, anchor="center")
        self._tree.column("name",     width=380, stretch=True,  anchor="w")
        self._tree.column("duration", width=80,  stretch=False, anchor="center")
        self._tree.column("size",     width=80,  stretch=False, anchor="e")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._tree.bind("<Button-1>", self._toggle_check)

        self._vid_status_var = tk.StringVar(value="")
        ttk.Label(vid_frame, textvariable=self._vid_status_var,
                  foreground="gray").grid(row=2, column=0, columnspan=3,
                                          sticky="w", pady=(2, 0))

        # ── Osztályok ─────────────────────────────────────────────────────────
        cls_frame = ttk.LabelFrame(self, text="Gyűjtendő osztályok", padding=8)
        cls_frame.pack(fill="x", **pad)

        self._cls_vars: dict[str, tk.BooleanVar] = {}
        for i, cls in enumerate(ALL_CLASSES):
            var = tk.BooleanVar(value=(cls in DEFAULT_TRIGGERS))
            self._cls_vars[cls] = var
            ttk.Checkbutton(cls_frame, text=cls, variable=var).grid(
                row=0, column=i, sticky="w", padx=12, pady=2)

        # ── Paraméterek ───────────────────────────────────────────────────────
        param_frame = ttk.LabelFrame(self, text="Paraméterek", padding=8)
        param_frame.pack(fill="x", **pad)

        self._frame_skip_var = tk.IntVar(value=5)
        self._max_per_id_var = tk.IntVar(value=1)
        self._conf_var       = tk.DoubleVar(value=0.35)
        self._device_var     = tk.StringVar(value="0")

        self._add_spin(param_frame, 0, 0, "Frame skip:",
                       self._frame_skip_var, 1, 30)
        self._add_spin(param_frame, 0, 3, "Max kép / jármű:",
                       self._max_per_id_var, 1, 10)
        self._add_spin_float(param_frame, 1, 0, "Konfidencia küszöb:",
                             self._conf_var, 0.1, 0.9, 0.05)

        ttk.Label(param_frame, text="Eszköz:").grid(
            row=1, column=3, sticky="e", padx=(20, 4))
        dev_inner = ttk.Frame(param_frame)
        dev_inner.grid(row=1, column=4, sticky="w")
        ttk.Radiobutton(dev_inner, text="GPU (0)", variable=self._device_var,
                        value="0").pack(side="left")
        ttk.Radiobutton(dev_inner, text="CPU", variable=self._device_var,
                        value="cpu").pack(side="left", padx=8)

        # ── Gombok ───────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)

        self._start_btn = ttk.Button(btn_frame, text="▶  Indítás",
                                     command=self._start, width=18)
        self._start_btn.pack(side="left", padx=(0, 8))

        self._stop_btn = ttk.Button(btn_frame, text="■  Leállítás",
                                    command=self._stop, width=18, state="disabled")
        self._stop_btn.pack(side="left")

        self._status_var = tk.StringVar(value="Kész.")
        ttk.Label(btn_frame, textvariable=self._status_var,
                  foreground="gray").pack(side="left", padx=16)

        # ── Log ───────────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Napló", padding=4)
        log_frame.pack(fill="both", expand=True, **pad)

        mono = font.Font(family="Consolas", size=9)
        self._log = scrolledtext.ScrolledText(
            log_frame, state="disabled", height=10,
            font=mono, background="#1e1e1e", foreground="#d4d4d4",
            insertbackground="white",
        )
        self._log.pack(fill="both", expand=True)

    # ── Segédmetódusok ────────────────────────────────────────────────────────

    def _add_path_row(self, parent, row, label, var, cmd):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="e", padx=(0, 6), pady=3)
        ttk.Entry(parent, textvariable=var, width=60).grid(
            row=row, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text="Tallóz…", command=cmd, width=9).grid(
            row=row, column=2, padx=(6, 0), pady=3)

    def _add_spin(self, parent, row, col, label, var, mn, mx):
        ttk.Label(parent, text=label).grid(
            row=row, column=col, sticky="e", padx=(0, 4), pady=4)
        ttk.Spinbox(parent, from_=mn, to=mx, textvariable=var,
                    width=6).grid(row=row, column=col + 1, sticky="w")

    def _add_spin_float(self, parent, row, col, label, var, mn, mx, inc):
        ttk.Label(parent, text=label).grid(
            row=row, column=col, sticky="e", padx=(0, 4), pady=4)
        ttk.Spinbox(parent, from_=mn, to=mx, increment=inc,
                    textvariable=var, format="%.2f",
                    width=6).grid(row=row, column=col + 1, sticky="w")

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # ── Videó scan ────────────────────────────────────────────────────────────

    def _scan_videos(self):
        path_str = self._video_var.get().strip()
        if not path_str:
            return
        for item in self._tree.get_children():
            self._tree.delete(item)
        videos = scan_videos(path_str)
        if not videos:
            self._vid_status_var.set("Nem található videó a megadott útvonalban.")
            return
        self._vid_status_var.set(f"Betöltés… ({len(videos)} videó)")
        self.update_idletasks()
        for v in videos:
            dur  = get_video_duration(v)
            size = f"{v.stat().st_size / 1_048_576:.0f} MB"
            self._tree.insert("", "end", iid=str(v),
                              values=(CHECK_ON, v.name, dur, size))
        self._vid_status_var.set(
            f"{len(videos)} videó  –  kattints a ☑/☐ ikonra a ki/bejelöléshez")

    def _toggle_check(self, event):
        col  = self._tree.identify_column(event.x)
        item = self._tree.identify_row(event.y)
        if col == "#1" and item:
            vals = list(self._tree.item(item, "values"))
            vals[0] = CHECK_OFF if vals[0] == CHECK_ON else CHECK_ON
            self._tree.item(item, values=vals)

    def _checked_videos(self) -> list[str]:
        return [
            iid for iid in self._tree.get_children()
            if self._tree.item(iid, "values")[0] == CHECK_ON
        ]

    # ── Tallóz ────────────────────────────────────────────────────────────────

    def _browse_video_file(self):
        path = filedialog.askopenfilename(
            title="Válassz videó fájlt",
            filetypes=[("Videó", "*.mp4 *.avi *.mov *.mkv *.mts *.m2ts *.wmv *.ts"),
                       ("Minden fájl", "*.*")],
            initialdir=self._video_var.get() or "/",
        )
        if path:
            self._video_var.set(path)

    def _browse_video_dir(self):
        path = filedialog.askdirectory(
            title="Válassz videó mappát",
            initialdir=self._video_var.get() or "/",
        )
        if path:
            self._video_var.set(path)

    def _browse_model(self):
        init = str(Path(self._model_var.get()).parent) if self._model_var.get() else "/"
        path = filedialog.askopenfilename(
            title="Válassz YOLO modell fájlt (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("Minden fájl", "*.*")],
            initialdir=init,
        )
        if path:
            self._model_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(
            title="Válassz kimeneti mappát",
            initialdir=self._output_var.get() or "/",
        )
        if path:
            self._output_var.set(path)

    # ── Indítás / leállítás ───────────────────────────────────────────────────

    def _start(self):
        triggers = [cls for cls, var in self._cls_vars.items() if var.get()]
        if not triggers:
            self._log_write("[HIBA] Legalább egy osztályt jelölj be!\n", "red")
            return
        checked = self._checked_videos()
        if not checked:
            self._log_write("[HIBA] Legalább egy videót jelölj be a listában!\n", "red")
            return
        model  = self._model_var.get().strip()
        output = self._output_var.get().strip()
        if not model:
            self._log_write("[HIBA] Adj meg modell fájlt!\n", "red")
            return
        if not output:
            self._log_write("[HIBA] Adj meg kimeneti mappát!\n", "red")
            return

        script = Path(__file__).parent / "extract_frames.py"
        python = sys.executable

        # Log törlése új futás előtt
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

        self._log_write("─" * 60 + "\n")
        self._log_write(f"[INFO] {len(checked)} videó feldolgozása\n\n")

        self._running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status_var.set("Fut…")

        threading.Thread(
            target=self._run_all,
            args=(checked, script, python, model, output, triggers),
            daemon=True,
        ).start()

    def _run_all(self, video_paths, script, python, model, output, triggers):
        for video in video_paths:
            if not self._running:
                break
            cmd = [
                python, str(script),
                "--video",      video,
                "--model",      model,
                "--output",     output,
                "--triggers",   ",".join(triggers),
                "--frame_skip", str(self._frame_skip_var.get()),
                "--max_per_id", str(self._max_per_id_var.get()),
                "--conf",       f"{self._conf_var.get():.2f}",
                "--device",     self._device_var.get(),
                "--quiet",
            ]
            self._log_write(f"▶ {Path(video).name}\n")
            self._run_one(cmd)

        if self._running:
            self._log_write("\n[KÉSZ] Minden videó feldolgozva.\n", "green")
            self.after(0, lambda: self._status_var.set("Kész."))
        else:
            self.after(0, lambda: self._status_var.set("Leállítva."))

        self._running = False
        self._process = None
        self.after(0, self._reset_buttons)

    def _run_one(self, cmd):
        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            # Bájtonként olvassuk, hogy a \r (tqdm progress) helyesen kezelje
            buf = b""
            for chunk in iter(lambda: self._process.stdout.read(128), b""):
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    cr = buf.find(b"\r")
                    if nl == -1 and cr == -1:
                        break
                    if nl != -1 and (cr == -1 or nl < cr):
                        # normál sor
                        line = buf[:nl + 1].decode("utf-8", errors="replace")
                        buf = buf[nl + 1:]
                        self._log_append(_ANSI_RE.sub("", line), overwrite=False)
                    else:
                        # \r → felülírjuk az utolsó sort
                        line = buf[:cr].decode("utf-8", errors="replace")
                        buf = buf[cr + 1:]
                        if line:
                            self._log_append(_ANSI_RE.sub("", line), overwrite=True)
            if buf:
                self._log_append(_ANSI_RE.sub("", buf.decode("utf-8", errors="replace")), overwrite=False)
            self._process.wait()
        except Exception as e:
            self._log_append(f"[KIVÉTEL] {e}\n", overwrite=False)

    def _stop(self):
        self._running = False
        if self._process:
            self._process.terminate()
            self._log_write("\n[INFO] Leállítás kérve…\n", "yellow")
            self._status_var.set("Leállítva.")

    def _reset_buttons(self):
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # ── Log írás (szálbiztos) ────────────────────────────────────────────────

    def _log_append(self, text: str, overwrite: bool = False, color: str | None = None):
        """Szálbiztos log írás. overwrite=True esetén felülírja az utolsó sort (\r)."""
        colors = {"red": "#f48771", "green": "#89d185", "yellow": "#dcdcaa"}

        def _do():
            self._log.config(state="normal")
            if overwrite:
                # Utolsó sor törlése (tqdm progress felülírás)
                self._log.delete("end-1l linestart", "end-1l lineend")
                self._log.insert("end-1l linestart", text)
            else:
                if color and color in colors:
                    tag = f"col_{color}"
                    self._log.tag_config(tag, foreground=colors[color])
                    self._log.insert("end", text, tag)
                else:
                    self._log.insert("end", text)
            # Auto-scroll ha a néző már lent van (utolsó 5%-ban)
            try:
                pos = self._log.yview()
                if pos[1] >= 0.95:
                    self._log.see("end")
            except Exception:
                pass
            self._log.config(state="disabled")
        self.after(0, _do)

    def _log_write(self, text: str, color: str | None = None):
        self._log_append(text, overwrite=False, color=color)


# ── Belépési pont ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
