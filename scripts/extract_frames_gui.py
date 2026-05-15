#!/usr/bin/env python3
"""
Frame Extractor – GUI indító
============================
Grafikus felület az extract_frames.py futtatásához.
Indítás:
    python scripts/extract_frames_gui.py
"""

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, scrolledtext, ttk

# ── Alapértelmezett útvonalak ─────────────────────────────────────────────────
DEFAULT_MODEL  = r"C:\Users\admin\Trafic_mojo_2\TM_modulok_py\Traffic_Mojo_2_0\yolo11s.pt"
DEFAULT_VIDEO  = r"C:\Users\admin\Trafic_mojo_2\Video"
DEFAULT_OUTPUT = r"C:\Users\admin\Trafic_mojo_2\YOLO-training\training_data\raw"

# COCO osztályok amiket érdemes gyűjteni (alapból bepipálva)
DEFAULT_TRIGGERS = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

# Összes releváns COCO osztály (amiből lehet választani)
ALL_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic_light", "stop_sign",
]


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Frame Extractor – képgyűjtő")
        self.resizable(True, True)
        self.minsize(700, 620)

        self._process: subprocess.Popen | None = None
        self._running = False

        self._build_ui()
        self._center_window()

    # ── UI felépítés ──────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # ── Fájl/mappa választók ──────────────────────────────────────────────
        paths_frame = ttk.LabelFrame(self, text="Útvonalak", padding=8)
        paths_frame.pack(fill="x", **pad)
        paths_frame.columnconfigure(1, weight=1)

        self._video_var  = tk.StringVar(value=DEFAULT_VIDEO)
        self._model_var  = tk.StringVar(value=DEFAULT_MODEL)
        self._output_var = tk.StringVar(value=DEFAULT_OUTPUT)

        self._add_path_row(paths_frame, 0, "Videó (fájl vagy mappa):",
                           self._video_var, self._browse_video)
        self._add_path_row(paths_frame, 1, "Modell (.pt):",
                           self._model_var, self._browse_model)
        self._add_path_row(paths_frame, 2, "Kimeneti mappa:",
                           self._output_var, self._browse_output)

        # ── Osztályok ─────────────────────────────────────────────────────────
        cls_frame = ttk.LabelFrame(self, text="Gyűjtendő osztályok", padding=8)
        cls_frame.pack(fill="x", **pad)

        self._cls_vars: dict[str, tk.BooleanVar] = {}
        for i, cls in enumerate(ALL_CLASSES):
            var = tk.BooleanVar(value=(cls in DEFAULT_TRIGGERS))
            self._cls_vars[cls] = var
            cb = ttk.Checkbutton(cls_frame, text=cls, variable=var)
            cb.grid(row=i // 4, column=i % 4, sticky="w", padx=10, pady=2)

        # ── Paraméterek ───────────────────────────────────────────────────────
        param_frame = ttk.LabelFrame(self, text="Paraméterek", padding=8)
        param_frame.pack(fill="x", **pad)

        self._frame_skip_var = tk.IntVar(value=5)
        self._max_per_id_var = tk.IntVar(value=1)
        self._conf_var       = tk.DoubleVar(value=0.35)
        self._device_var     = tk.StringVar(value="0")

        row = 0
        self._add_spin(param_frame, row, 0, "Frame skip (minden N. frame):",
                       self._frame_skip_var, 1, 30)
        self._add_spin(param_frame, row, 3, "Max kép / jármű:",
                       self._max_per_id_var, 1, 10)
        row += 1
        self._add_spin_float(param_frame, row, 0, "Konfidencia küszöb (0–1):",
                             self._conf_var, 0.1, 0.9, 0.05)

        ttk.Label(param_frame, text="Eszköz:").grid(
            row=row, column=3, sticky="e", padx=(20, 4))
        dev_inner = ttk.Frame(param_frame)
        dev_inner.grid(row=row, column=4, sticky="w")
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
            log_frame, state="disabled", height=14,
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

    def _add_spin(self, parent, row, col_offset, label, var, mn, mx):
        ttk.Label(parent, text=label).grid(
            row=row, column=col_offset, sticky="e", padx=(0, 4), pady=4)
        ttk.Spinbox(parent, from_=mn, to=mx, textvariable=var,
                    width=6).grid(row=row, column=col_offset + 1, sticky="w")

    def _add_spin_float(self, parent, row, col_offset, label, var, mn, mx, inc):
        ttk.Label(parent, text=label).grid(
            row=row, column=col_offset, sticky="e", padx=(0, 4), pady=4)
        ttk.Spinbox(parent, from_=mn, to=mx, increment=inc,
                    textvariable=var, format="%.2f",
                    width=6).grid(row=row, column=col_offset + 1, sticky="w")

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # ── Tallóz ────────────────────────────────────────────────────────────────

    def _browse_video(self):
        # Először fájlt próbálunk, ha Cancel → mappát
        path = filedialog.askopenfilename(
            title="Válassz videó fájlt (vagy Cancel a mappa-választóhoz)",
            filetypes=[("Videó", "*.mp4 *.avi *.mov *.mkv *.mts *.m2ts *.wmv *.ts"),
                       ("Minden fájl", "*.*")],
            initialdir=self._video_var.get() or "/",
        )
        if not path:
            path = filedialog.askdirectory(
                title="Válassz videó mappát",
                initialdir=self._video_var.get() or "/",
            )
        if path:
            self._video_var.set(path)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Válassz YOLO modell fájlt (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("Minden fájl", "*.*")],
            initialdir=str(Path(self._model_var.get()).parent),
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

        video  = self._video_var.get().strip()
        model  = self._model_var.get().strip()
        output = self._output_var.get().strip()

        if not video:
            self._log_write("[HIBA] Adj meg videó fájlt vagy mappát!\n", "red")
            return
        if not model:
            self._log_write("[HIBA] Adj meg modell fájlt!\n", "red")
            return
        if not output:
            self._log_write("[HIBA] Adj meg kimeneti mappát!\n", "red")
            return

        script = Path(__file__).parent / "extract_frames.py"
        python = sys.executable

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
        ]

        self._log_write("─" * 60 + "\n")
        self._log_write("Parancs: " + " ".join(cmd) + "\n\n")

        self._running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status_var.set("Fut…")

        threading.Thread(target=self._run, args=(cmd,), daemon=True).start()

    def _run(self, cmd):
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in self._process.stdout:
                self._log_write(line)
            self._process.wait()
            rc = self._process.returncode
            if rc == 0:
                self._log_write("\n[KÉSZ] Feldolgozás befejezve.\n", "green")
                self.after(0, lambda: self._status_var.set("Kész."))
            else:
                self._log_write(f"\n[LEÁLLT] Visszatérési kód: {rc}\n", "red")
                self.after(0, lambda: self._status_var.set(f"Hiba (kód: {rc})"))
        except Exception as e:
            self._log_write(f"\n[KIVÉTEL] {e}\n", "red")
            self.after(0, lambda: self._status_var.set("Hiba"))
        finally:
            self._running = False
            self._process = None
            self.after(0, self._reset_buttons)

    def _stop(self):
        if self._process and self._running:
            self._process.terminate()
            self._log_write("\n[INFO] Leállítás kérve…\n", "yellow")
            self._status_var.set("Leállítva.")

    def _reset_buttons(self):
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # ── Log írás (szálbiztos) ─────────────────────────────────────────────────

    def _log_write(self, text: str, color: str | None = None):
        def _do():
            self._log.config(state="normal")
            if color:
                tag = f"col_{color}"
                colors = {"red": "#f48771", "green": "#89d185",
                          "yellow": "#dcdcaa", "white": "#d4d4d4"}
                self._log.tag_config(tag, foreground=colors.get(color, "#d4d4d4"))
                self._log.insert("end", text, tag)
            else:
                self._log.insert("end", text)
            self._log.see("end")
            self._log.config(state="disabled")
        self.after(0, _do)


# ── Belépési pont ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
