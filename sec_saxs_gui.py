"""
SEC-SAXS GUI - interactive version of SEC_SAXS_main_user_v1.ipynb

Workflow:
  1. Choose the rebinned data folder and one or two q-ranges, then press
     "Load data" -> chromatogram (average intensity vs. frame) is plotted.
  2. Select the buffer region (drag on the chromatogram while "Buffer" is
     active, or type the start/end frames).
  3. Select the sample region the same way.
  4. Optionally set a buffer scaling factor, then press
     "Calculate scattering pattern":
        I_sub   = I_sample - s * I_buffer
        err_sub = sqrt(err_sample^2 + (s * err_buffer)^2)
  5. Save the chromatogram and/or the merged/subtracted curves.

Run:  python sec_saxs_gui.py
"""

import os
import re
import glob
import queue
import threading

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # keep output folders (SAXS_output, output_chromatogram) where the notebook puts them

from functions import save_data, save_chromatogram  # noqa: E402

BUFFER_COLOR = "#1f77b4"
SAMPLE_COLOR = "#ff7f0e"
SUB_COLOR = "#2ca02c"


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------
class SECData:
    """All frames of one scan held in memory: I[frame, q], err[frame, q]."""

    def __init__(self, folder, suffix, frames, q, I, err):
        self.folder = folder
        self.suffix = suffix
        self.frames = frames  # 1D int array of frame numbers
        self.q = q            # 1D
        self.I = I            # 2D (n_frames, n_q)
        self.err = err        # 2D (n_frames, n_q)

    @property
    def scan(self):
        return os.path.basename(os.path.normpath(self.folder)).split("_")[0]

    def chromatogram(self, qmin, qmax):
        mask = (self.q >= qmin) & (self.q <= qmax)
        if not mask.any():
            raise ValueError(f"No q-points between {qmin} and {qmax}")
        return np.nanmean(self.I[:, mask], axis=1)

    def merge(self, start, end):
        """Average frames start..end (inclusive), same formulas as functions.merge_frames."""
        sel = (self.frames >= start) & (self.frames <= end)
        n = int(sel.sum())
        if n == 0:
            raise ValueError(f"No frames between {start} and {end}")
        I_m = self.I[sel].mean(axis=0)
        err_m = np.sqrt((self.err[sel] ** 2).sum(axis=0)) / n
        return self.q, I_m, err_m, n


def list_frame_files(folder, suffix):
    pattern = re.compile(r"shot_(\d+)" + re.escape(suffix) + r"\.dat$")
    files = []
    for path in glob.glob(os.path.join(folder, f"shot_*{suffix}.dat")):
        m = pattern.search(os.path.basename(path))
        if m:
            files.append((int(m.group(1)), path))
    files.sort()
    return files


def load_folder(folder, suffix, progress_cb):
    files = list_frame_files(folder, suffix)
    if not files:
        raise FileNotFoundError(f"No files matching shot_XXXX{suffix}.dat in\n{folder}")

    frames, Is, errs = [], [], []
    q_ref = None
    for i, (frame, path) in enumerate(files):
        arr = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None).to_numpy(dtype=float)
        q = arr[:, 0]
        if q_ref is None:
            q_ref = q
        elif len(q) != len(q_ref):
            raise ValueError(f"Frame {frame} has {len(q)} q-points, expected {len(q_ref)}.\n"
                             "All frames must share the same (rebinned) q-grid.")
        frames.append(frame)
        Is.append(arr[:, 1])
        errs.append(arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(q))
        if i % 25 == 0 or i == len(files) - 1:
            progress_cb(i + 1, len(files))

    return SECData(folder, suffix, np.array(frames), q_ref, np.vstack(Is), np.vstack(errs))


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class SECSAXSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SEC-SAXS buffer subtraction")
        self.geometry("1400x900")

        self.data = None
        self.chroms = {}        # (qmin, qmax) -> chromatogram array
        self.result = None      # dict with merged / subtracted curves
        self.selectors = []
        self.span_patches = []
        self.chrom_axes = []
        self._load_queue = queue.Queue()

        self._build_controls()
        self._build_plots()

    # ---------------- layout ----------------
    def _build_controls(self):
        panel = ttk.Frame(self, padding=8)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        # --- data folder
        box = ttk.LabelFrame(panel, text="Data", padding=6)
        box.pack(fill=tk.X, pady=4)
        default_parent = "rebinned_output"
        self.parent_var = tk.StringVar(value=default_parent if os.path.isdir(default_parent) else "")
        ttk.Label(box, text="Parent folder:").grid(row=0, column=0, sticky="w")
        parent_entry = ttk.Entry(box, textvariable=self.parent_var, width=30)
        parent_entry.grid(row=1, column=0, sticky="we")
        parent_entry.bind("<Return>", lambda e: self._refresh_subfolders())
        parent_entry.bind("<FocusOut>", lambda e: self._refresh_subfolders())
        ttk.Button(box, text="Browse…", command=self._browse).grid(row=1, column=1, padx=2)

        ttk.Label(box, text="Scan subfolder:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.sub_var = tk.StringVar()
        self.sub_combo = ttk.Combobox(box, textvariable=self.sub_var, width=28)
        self.sub_combo.grid(row=3, column=0, sticky="we")
        ttk.Button(box, text="↻", width=3, command=self._refresh_subfolders).grid(row=3, column=1, padx=2)

        ttk.Label(box, text="File suffix:").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.suffix_var = tk.StringVar(value="_rebinned")
        ttk.Entry(box, textvariable=self.suffix_var, width=14).grid(row=5, column=0, sticky="w")
        self._refresh_subfolders()

        # --- q ranges
        box = ttk.LabelFrame(panel, text="1. q-range(s) for average intensity (1/Å)", padding=6)
        box.pack(fill=tk.X, pady=4)
        ttk.Label(box, text="q min").grid(row=0, column=1)
        ttk.Label(box, text="q max").grid(row=0, column=2)
        ttk.Label(box, text="Range 1").grid(row=1, column=0, sticky="w")
        self.qmin1 = tk.StringVar(value="0.01")
        self.qmax1 = tk.StringVar(value="0.08")
        ttk.Entry(box, textvariable=self.qmin1, width=9).grid(row=1, column=1, padx=2)
        ttk.Entry(box, textvariable=self.qmax1, width=9).grid(row=1, column=2, padx=2)

        self.use_q2 = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Range 2", variable=self.use_q2,
                        command=self._toggle_q2).grid(row=2, column=0, sticky="w")
        self.qmin2 = tk.StringVar(value="0.003")
        self.qmax2 = tk.StringVar(value="0.01")
        self.q2_entries = [ttk.Entry(box, textvariable=self.qmin2, width=9),
                           ttk.Entry(box, textvariable=self.qmax2, width=9)]
        self.q2_entries[0].grid(row=2, column=1, padx=2)
        self.q2_entries[1].grid(row=2, column=2, padx=2)

        self.load_btn = ttk.Button(box, text="Load data + plot chromatogram", command=self._load)
        self.load_btn.grid(row=3, column=0, columnspan=3, sticky="we", pady=(6, 0))
        ttk.Button(box, text="Update chromatogram (new q-ranges)",
                   command=self._update_chromatogram).grid(row=4, column=0, columnspan=3, sticky="we", pady=2)
        self.progress = ttk.Progressbar(box, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky="we", pady=2)

        # --- region selection
        box = ttk.LabelFrame(panel, text="2-3. Frame ranges (drag on chromatogram)", padding=6)
        box.pack(fill=tk.X, pady=4)
        self.drag_target = "sample"
        ttk.Label(box, text="Click + drag = Sample\nShift + click + drag = Buffer",
                  foreground="gray25").grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(box, text="start").grid(row=1, column=1)
        ttk.Label(box, text="end").grid(row=1, column=2)
        self.buf_start = tk.StringVar(value="1250")
        self.buf_end = tk.StringVar(value="1400")
        self.smp_start = tk.StringVar(value="860")
        self.smp_end = tk.StringVar(value="920")
        tk.Label(box, text="Buffer", fg=BUFFER_COLOR).grid(row=2, column=0, sticky="w")
        tk.Label(box, text="Sample", fg=SAMPLE_COLOR).grid(row=3, column=0, sticky="w")
        self.frame_spins = []
        for row, (a, b) in ((2, (self.buf_start, self.buf_end)), (3, (self.smp_start, self.smp_end))):
            for col, var in ((1, a), (2, b)):
                sp = ttk.Spinbox(box, from_=0, to=100000, increment=1, textvariable=var, width=8,
                                 command=self._draw_spans)
                sp.grid(row=row, column=col, padx=2, pady=1)
                sp.bind("<Return>", lambda e: self._draw_spans())
                sp.bind("<FocusOut>", lambda e: self._draw_spans())
                self.frame_spins.append(sp)
        self.count_label = ttk.Label(box, text="")
        self.count_label.grid(row=4, column=0, columnspan=3, sticky="w")

        # --- scaling + calculate
        box = ttk.LabelFrame(panel, text="4-5. Scattering pattern", padding=6)
        box.pack(fill=tk.X, pady=4)
        ttk.Label(box, text="Buffer scale factor:").grid(row=0, column=0, sticky="w")
        self.scale_var = tk.StringVar(value="1.0")
        sc = ttk.Entry(box, textvariable=self.scale_var, width=9)
        sc.grid(row=0, column=1, padx=2)
        sc.bind("<Return>", lambda e: self._calculate())
        ttk.Button(box, text="−", width=3,
                   command=lambda: self._step_scale(-0.001)).grid(row=0, column=2, padx=1)
        ttk.Button(box, text="+", width=3,
                   command=lambda: self._step_scale(0.001)).grid(row=0, column=3, padx=1)
        self.show_raw = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Show buffer & sample curves", variable=self.show_raw,
                        command=self._plot_scattering).grid(row=1, column=0, columnspan=4, sticky="w")
        ttk.Button(box, text="Calculate scattering pattern",
                   command=self._calculate).grid(row=2, column=0, columnspan=4, sticky="we", pady=(6, 0))

        # --- save
        box = ttk.LabelFrame(panel, text="Save", padding=6)
        box.pack(fill=tk.X, pady=4)
        ttk.Label(box, text="Sample name:").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar(value="sample_1")
        ttk.Entry(box, textvariable=self.name_var, width=14).grid(row=0, column=1, padx=2)
        ttk.Label(box, text="Buffer name:").grid(row=1, column=0, sticky="w")
        self.bufname_var = tk.StringVar(value="buffer_1")
        ttk.Entry(box, textvariable=self.bufname_var, width=14).grid(row=1, column=1, padx=2)
        ttk.Button(box, text="Save chromatogram(s)",
                   command=self._save_chromatogram).grid(row=2, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Button(box, text="Save buffer / sample / subtracted",
                   command=self._save_curves).grid(row=3, column=0, columnspan=2, sticky="we", pady=2)

        self.status = tk.StringVar(value="Choose folder and q-range(s), then press Load.")
        ttk.Label(panel, textvariable=self.status, wraplength=300,
                  foreground="gray25").pack(fill=tk.X, pady=8)

    def _build_plots(self):
        right = ttk.PanedWindow(self, orient=tk.VERTICAL)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top = ttk.Frame(right)
        bottom = ttk.Frame(right)
        right.add(top, weight=1)
        right.add(bottom, weight=1)

        self.fig_chrom = Figure(figsize=(9, 4), constrained_layout=True)
        self.canvas_chrom = FigureCanvasTkAgg(self.fig_chrom, master=top)
        NavigationToolbar2Tk(self.canvas_chrom, top).update()
        self.canvas_chrom.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_chrom.mpl_connect("button_press_event", self._on_press)

        self.fig_sas = Figure(figsize=(9, 4), constrained_layout=True)
        self.ax_sas = self.fig_sas.add_subplot(111)
        self.canvas_sas = FigureCanvasTkAgg(self.fig_sas, master=bottom)
        NavigationToolbar2Tk(self.canvas_sas, bottom).update()
        self.canvas_sas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._style_sas_axes()
        self.canvas_sas.draw_idle()

    # ---------------- helpers ----------------
    def _toggle_q2(self):
        state = "normal" if self.use_q2.get() else "disabled"
        for e in self.q2_entries:
            e.configure(state=state)

    def _browse(self):
        start = self.parent_var.get() if os.path.isdir(self.parent_var.get()) else SCRIPT_DIR
        d = filedialog.askdirectory(initialdir=start, title="Select parent folder (e.g. rebinned_output)")
        if d:
            self.parent_var.set(d)
            self._refresh_subfolders()

    def _refresh_subfolders(self):
        parent = self.parent_var.get().strip()
        subs = sorted(d for d in os.listdir(parent)
                      if os.path.isdir(os.path.join(parent, d))) if os.path.isdir(parent) else []
        self.sub_combo["values"] = subs
        if self.sub_var.get() not in subs:
            self.sub_var.set(subs[0] if subs else "")

    def _folder(self):
        return os.path.join(self.parent_var.get().strip(), self.sub_var.get().strip())

    def _q_ranges(self):
        pairs = [(self.qmin1.get(), self.qmax1.get())]
        if self.use_q2.get():
            pairs.append((self.qmin2.get(), self.qmax2.get()))
        out = []
        for lo, hi in pairs:
            lo, hi = float(lo), float(hi)
            if lo >= hi:
                raise ValueError(f"q min ({lo}) must be smaller than q max ({hi})")
            out.append((lo, hi))
        return out

    def _frame_range(self, which):
        a, b = (self.buf_start, self.buf_end) if which == "buffer" else (self.smp_start, self.smp_end)
        lo, hi = int(float(a.get())), int(float(b.get()))
        return (lo, hi) if lo <= hi else (hi, lo)

    # ---------------- loading ----------------
    def _load(self):
        try:
            self._q_ranges()
        except ValueError as e:
            messagebox.showerror("Invalid q-range", str(e))
            return
        folder, suffix = self._folder(), self.suffix_var.get()
        if not self.sub_var.get().strip() or not os.path.isdir(folder):
            messagebox.showerror("Folder not found", f"Folder does not exist:\n{folder}")
            return

        self.load_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.status.set("Loading frames…")

        def progress(i, n):
            self._load_queue.put(("progress", i, n))

        def worker():
            try:
                data = load_folder(folder, suffix, progress)
                self._load_queue.put(("done", data))
            except Exception as e:  # noqa: BLE001
                self._load_queue.put(("error", e))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll_load)

    def _poll_load(self):
        try:
            while True:
                msg = self._load_queue.get_nowait()
                if msg[0] == "progress":
                    _, i, n = msg
                    self.progress["maximum"] = n
                    self.progress["value"] = i
                    self.status.set(f"Loading frames… {i}/{n}")
                elif msg[0] == "done":
                    self.data = msg[1]
                    self.load_btn.configure(state="normal")
                    for sp in self.frame_spins:
                        sp.configure(from_=int(self.data.frames.min()), to=int(self.data.frames.max()))
                    self.status.set(f"Scan {self.data.scan}: loaded {len(self.data.frames)} frames, "
                                    f"{len(self.data.q)} q-points "
                                    f"({self.data.q.min():.4g}–{self.data.q.max():.4g} 1/Å).")
                    self._update_chromatogram()
                    return
                elif msg[0] == "error":
                    self.load_btn.configure(state="normal")
                    self.status.set("Loading failed.")
                    messagebox.showerror("Loading failed", str(msg[1]))
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll_load)

    # ---------------- chromatogram ----------------
    def _update_chromatogram(self):
        if self.data is None:
            messagebox.showinfo("No data", "Load data first.")
            return
        try:
            ranges = self._q_ranges()
            self.chroms = {r: self.data.chromatogram(*r) for r in ranges}
        except ValueError as e:
            messagebox.showerror("Chromatogram", str(e))
            return

        for s in self.selectors:
            s.disconnect_events()
        self.selectors, self.span_patches = [], []
        self.fig_chrom.clear()

        n = len(ranges)
        self.chrom_axes = []
        for i, (qmin, qmax) in enumerate(ranges):
            ax = self.fig_chrom.add_subplot(n, 1, i + 1, sharex=self.chrom_axes[0] if self.chrom_axes else None)
            ax.plot(self.data.frames, self.chroms[(qmin, qmax)], color="k", lw=1)
            ax.set_yscale("log")
            ax.minorticks_on()
            ax.set_ylabel(f"<I> {qmin}<q<{qmax}")
            if i == n - 1:
                ax.set_xlabel("Frame (elution time in s at 1 frame/s)")
            else:
                ax.tick_params(labelbottom=False)
            self.chrom_axes.append(ax)
            self.selectors.append(SpanSelector(ax, self._on_span, "horizontal", useblit=True,
                                               minspan=0.5, props=dict(alpha=0.2, facecolor="gray")))
        self._draw_spans()

    def _on_press(self, event):
        # Decide at mouse-press time which region the drag will set
        shift = bool(event.key) and "shift" in event.key
        self.drag_target = "buffer" if shift else "sample"
        color = BUFFER_COLOR if shift else SAMPLE_COLOR
        for s in self.selectors:  # tint the live drag rectangle to match
            for artist in getattr(s, "artists", []):
                artist.set_facecolor(color)

    def _on_span(self, xmin, xmax):
        lo, hi = int(round(xmin)), int(round(xmax))
        if self.drag_target == "buffer":
            self.buf_start.set(str(lo)); self.buf_end.set(str(hi))
        else:
            self.smp_start.set(str(lo)); self.smp_end.set(str(hi))
        self._draw_spans()

    def _draw_spans(self):
        for p in self.span_patches:
            try:
                p.remove()
            except ValueError:
                pass
        self.span_patches = []
        texts = []
        for which, color in (("buffer", BUFFER_COLOR), ("sample", SAMPLE_COLOR)):
            try:
                lo, hi = self._frame_range(which)
            except ValueError:
                continue
            for k, ax in enumerate(self.chrom_axes):
                lab = which.capitalize() if k == 0 else None
                self.span_patches.append(ax.axvspan(lo, hi + 1, color=color, alpha=0.25, label=lab))
            if self.data is not None:
                n = int(((self.data.frames >= lo) & (self.data.frames <= hi)).sum())
                texts.append(f"{which}: {n} frames")
        if self.chrom_axes:
            self.chrom_axes[0].legend(loc="upper right")
            self.canvas_chrom.draw_idle()
        self.count_label.configure(text=", ".join(texts))

    # ---------------- scattering ----------------
    def _step_scale(self, delta):
        try:
            value = float(self.scale_var.get())
        except ValueError:
            value = 1.0
        self.scale_var.set(f"{round(value + delta, 6):g}")
        if self.data is not None:
            self._calculate()

    def _calculate(self):
        if self.data is None:
            messagebox.showinfo("No data", "Load data first.")
            return
        try:
            scale = float(self.scale_var.get())
            b0, b1 = self._frame_range("buffer")
            s0, s1 = self._frame_range("sample")
            bq, bI, be, bn = self.data.merge(b0, b1)
            sq, sI, se, sn = self.data.merge(s0, s1)
        except ValueError as e:
            messagebox.showerror("Calculation", str(e))
            return

        subI = sI - scale * bI
        sube = np.sqrt(se ** 2 + (scale * be) ** 2)
        self.result = dict(q=sq, buffer=(bI, be, b0, b1, bn), sample=(sI, se, s0, s1, sn),
                           sub=(subI, sube), scale=scale, scan=self.data.scan)
        self.status.set(f"Merged {bn} buffer frames ({b0}–{b1}) and {sn} sample frames ({s0}–{s1}); "
                        f"buffer scale = {scale:g}.")
        self._plot_scattering()

    def _style_sas_axes(self):
        ax = self.ax_sas
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("q (1/Å)")
        ax.set_ylabel("I (1/cm)")

    def _plot_scattering(self):
        if self.result is None:
            return
        r = self.result
        ax = self.ax_sas
        ax.clear()
        q = r["q"]
        if self.show_raw.get():
            bI, be, b0, b1, _ = r["buffer"]
            sI, se, s0, s1, _ = r["sample"]
            blabel = f"Buffer {b0}–{b1}" + (f" × {r['scale']:g}" if r["scale"] != 1 else "")
            ax.errorbar(q, r["scale"] * bI, yerr=r["scale"] * be, fmt="o", ms=2, color=BUFFER_COLOR,
                        label=blabel)
            ax.errorbar(q, sI, yerr=se, fmt="o", ms=2, color=SAMPLE_COLOR, label=f"Sample {s0}–{s1}")
        subI, sube = r["sub"]
        ax.errorbar(q, subI, yerr=sube, fmt="o", ms=2, color=SUB_COLOR, label="Subtracted")
        self._style_sas_axes()
        ax.set_title(f"Scan {r['scan']}")
        ax.legend()
        self.canvas_sas.draw_idle()

    # ---------------- saving ----------------
    def _save_chromatogram(self):
        if self.data is None or not self.chroms:
            messagebox.showinfo("Nothing to save", "Plot a chromatogram first.")
            return
        for (qmin, qmax), chrom in self.chroms.items():
            save_chromatogram(self.data.scan, qmin, qmax, self.data.frames, chrom)
        self.status.set(f"Saved {len(self.chroms)} chromatogram(s) to output_chromatogram/.")

    def _save_curves(self):
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Calculate the scattering pattern first.")
            return
        r = self.result
        scan, q = r["scan"], r["q"]
        name, bname = self.name_var.get().strip() or "sample", self.bufname_var.get().strip() or "buffer"
        bI, be, b0, b1, _ = r["buffer"]
        sI, se, s0, s1, _ = r["sample"]
        subI, sube = r["sub"]
        sub_name = f"{name}_subtracted_{bname}"
        if r["scale"] != 1:
            sub_name += f"_scale{r['scale']:g}"
        save_data(scan, name, s0, s1, q, sI, se)
        save_data(scan, bname, b0, b1, q, bI, be)
        save_data(scan, sub_name, s0, s1, q, subI, sube)
        self.status.set(f"Saved {name}, {bname} and {sub_name} to SAXS_output/.")


if __name__ == "__main__":
    SECSAXSApp().mainloop()
