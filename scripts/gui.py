"""Tkinter control panel for scripts/blender_animate.py.

Lets you pick a GPX route, inspect/edit its flight plan, choose preview vs.
full-animation rendering, launch Blender headless, and watch progress live
(parsed from Blender's own stdout) plus browse the generated preview images.

Run with the project venv (needs gpxpy + tkinter, both already in .venv):
    .venv\\Scripts\\python.exe scripts\\gui.py
"""

import dataclasses
import importlib.util
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import gpxpy

BASE_DIR         = Path(__file__).resolve().parent.parent
SCRIPTS_DIR      = BASE_DIR / 'scripts'
GPX_DIR          = BASE_DIR / 'gpx'
FLIGHT_PLANS_DIR = BASE_DIR / 'flight_plans'
RENDER_DIR       = BASE_DIR / 'render'
ANIMATE_SCRIPT   = SCRIPTS_DIR / 'blender_animate.py'
FRAMES_TO_MP4_SCRIPT = SCRIPTS_DIR / 'frames_to_mp4.py'
TEMPLATE_PLAN    = FLIGHT_PLANS_DIR / '_template.py'

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))  # flight_plans/*.py do `from flight_plan import ...`

THUMB_WIDTH = 160
GALLERY_COLUMNS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers independent of Blender/bpy
# ─────────────────────────────────────────────────────────────────────────────

def list_gpx_names():
    return sorted(p.stem for p in GPX_DIR.glob('*.gpx'))


def flight_plan_path_for(gpx_name):
    return FLIGHT_PLANS_DIR / f'{gpx_name}.py'


def load_plan(gpx_name):
    """Import flight_plans/<gpx_name>.py the same way blender_animate.py does."""
    plan_path = flight_plan_path_for(gpx_name)
    if not plan_path.exists():
        raise FileNotFoundError(str(plan_path))
    spec = importlib.util.spec_from_file_location(f'flight_plans.{gpx_name}', plan_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PLAN


def summarize_step(step):
    fields = ', '.join(f'{f.name}={getattr(step, f.name)!r}' for f in dataclasses.fields(step))
    return f'{type(step).__name__}({fields})'


def count_preview_images(plan):
    """Mirror FlightPlan.preview_poses: 3 images per non-zero-duration step."""
    total = 0
    prev_end_t = 0.0
    for step in plan.steps:
        if step.end_t - prev_end_t >= 1e-6:
            total += 3
        prev_end_t = step.end_t
    return total


def read_int_constant(name, default):
    """Read a module-level `NAME = <int>` constant straight from blender_animate.py's
    source, so the frame-count estimate here never drifts from the real pipeline."""
    text = ANIMATE_SCRIPT.read_text(encoding='utf-8')
    m = re.search(rf'^{name}\s*=\s*(\d+)', text, re.MULTILINE)
    return int(m.group(1)) if m else default


def estimate_total_frames(gpx_path):
    """Replicates blender_animate.py's video-duration → frame-count calculation
    using only the GPX timestamps, without touching Blender."""
    fps   = read_int_constant('ANIMATION_FPS', 24)
    speed = read_int_constant('ANIMATION_SPEED', 600)

    with open(gpx_path, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)

    timestamps = [
        point.time
        for track in gpx.tracks
        for segment in track.segments
        for point in segment.points
    ]
    valid = [t for t in timestamps if t is not None]
    if len(valid) >= 2:
        seconds = (valid[-1] - valid[0]).total_seconds() / speed
    else:
        seconds = 60.0
    return max(1, int(seconds * fps))


def format_eta(seconds):
    if seconds is None or seconds != seconds or seconds < 0:  # NaN/negative guard
        return '--:--'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:d}:{s:02d}'


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('GPX Relief Tracer — Render Control')
        self.geometry('900x800')
        self.minsize(700, 600)

        self.proc = None
        self.log_queue = queue.Queue()
        self.reader_thread = None
        self.thumbnails = []  # keep references so PhotoImage isn't GC'd

        self.progress_mode = None       # 'preview' | 'animation' | 'video'
        self.progress_total = 0
        self.progress_start_value = 0   # ANIMATION_START_FRAME, for animation mode
        self.progress_current = 0
        self.first_progress_time = None
        self.first_progress_value = None

        self._build_ui()
        self._on_gpx_selected()
        self.after(100, self._poll_log_queue)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}

        main_pane = ttk.PanedWindow(self, orient='horizontal')
        main_pane.pack(fill='both', expand=True)

        left = ttk.Frame(main_pane)
        right = ttk.Frame(main_pane)
        main_pane.add(left, weight=2)
        main_pane.add(right, weight=1)

        # GPX selector
        gpx_frame = ttk.LabelFrame(left, text='GPX route')
        gpx_frame.pack(fill='x', **pad)
        self.gpx_var = tk.StringVar()
        self.gpx_combo = ttk.Combobox(gpx_frame, textvariable=self.gpx_var,
                                       values=list_gpx_names(), state='readonly')
        self.gpx_combo.pack(fill='x', padx=8, pady=8)
        self.gpx_combo.bind('<<ComboboxSelected>>', lambda e: self._on_gpx_selected())

        # Flight plan section
        plan_frame = ttk.LabelFrame(left, text='Flight plan')
        plan_frame.pack(fill='both', **pad)

        self.plan_path_label = ttk.Label(plan_frame, text='')
        self.plan_path_label.pack(anchor='w', padx=8, pady=(8, 0))

        plan_btns = ttk.Frame(plan_frame)
        plan_btns.pack(fill='x', padx=8, pady=4)
        self.create_plan_btn = ttk.Button(plan_btns, text='Create from template',
                                           command=self._create_plan_from_template)
        self.create_plan_btn.pack(side='left')
        ttk.Button(plan_btns, text='Open in editor', command=self._open_plan_in_editor).pack(side='left', padx=6)
        ttk.Button(plan_btns, text='Reload', command=self._on_gpx_selected).pack(side='left')

        self.plan_steps_list = tk.Listbox(plan_frame, height=6)
        self.plan_steps_list.pack(fill='both', expand=False, padx=8, pady=(0, 8))

        # Run options
        opts_frame = ttk.LabelFrame(left, text='Run options')
        opts_frame.pack(fill='x', **pad)

        row = ttk.Frame(opts_frame)
        row.pack(fill='x', padx=8, pady=8)
        ttk.Label(row, text='Start frame (ANIMATION_START_FRAME):').pack(side='left')
        self.start_frame_var = tk.StringVar(value='0')
        ttk.Entry(row, textvariable=self.start_frame_var, width=10).pack(side='left', padx=8)

        self.preview_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text='Preview only (RENDER_PREVIEW, skips full animation)',
                         variable=self.preview_var).pack(side='left', padx=20)

        # Render controls
        run_frame = ttk.Frame(left)
        run_frame.pack(fill='x', **pad)
        self.render_btn = ttk.Button(run_frame, text='Render', command=self._start_render)
        self.render_btn.pack(side='left')
        self.video_btn = ttk.Button(run_frame, text='Generate Video', command=self._start_generate_video)
        self.video_btn.pack(side='left', padx=8)
        self.cancel_btn = ttk.Button(run_frame, text='Cancel', command=self._cancel_render, state='disabled')
        self.cancel_btn.pack(side='left')

        # Progress
        progress_frame = ttk.Frame(left)
        progress_frame.pack(fill='x', **pad)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
        self.progress_bar.pack(fill='x')
        self.progress_label = ttk.Label(progress_frame, text='Idle')
        self.progress_label.pack(anchor='w', pady=(4, 0))

        # Log
        log_frame = ttk.LabelFrame(left, text='Log')
        log_frame.pack(fill='both', expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state='disabled', wrap='none')
        self.log_text.pack(fill='both', expand=True, padx=8, pady=8)

        # Gallery — kept on its own side pane so it stays visible on small screens
        gallery_frame = ttk.LabelFrame(right, text='Flight preview gallery')
        gallery_frame.pack(fill='both', expand=True, **pad)
        gallery_top = ttk.Frame(gallery_frame)
        gallery_top.pack(fill='x', padx=8, pady=(8, 0))
        ttk.Button(gallery_top, text='Refresh gallery', command=self._refresh_gallery).pack(side='left')

        gallery_canvas_holder = ttk.Frame(gallery_frame)
        gallery_canvas_holder.pack(fill='both', expand=True, padx=8, pady=8)
        self.gallery_canvas = tk.Canvas(gallery_canvas_holder)
        gallery_scroll = ttk.Scrollbar(gallery_canvas_holder, orient='vertical',
                                        command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=gallery_scroll.set)
        gallery_scroll.pack(side='right', fill='y')
        self.gallery_canvas.pack(side='left', fill='both', expand=True)
        self.gallery_inner = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor='nw')
        self.gallery_inner.bind('<Configure>',
                                 lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox('all')))

    # ── GPX / flight plan wiring ─────────────────────────────────────────

    def _on_gpx_selected(self):
        names = list_gpx_names()
        if not names:
            messagebox.showerror('No GPX files', f'No .gpx files found in {GPX_DIR}')
            return
        if self.gpx_var.get() not in names:
            self.gpx_var.set(names[0])

        gpx_name = self.gpx_var.get()
        plan_path = flight_plan_path_for(gpx_name)
        self.plan_path_label.config(text=str(plan_path))
        self.plan_steps_list.delete(0, 'end')

        if not plan_path.exists():
            self.plan_steps_list.insert('end', '(no flight plan file yet)')
            self.create_plan_btn.config(state='normal')
        else:
            self.create_plan_btn.config(state='disabled')
            try:
                plan = load_plan(gpx_name)
                self.plan_steps_list.insert('end', f'smoothing = {plan.smoothing}')
                for step in plan.steps:
                    self.plan_steps_list.insert('end', summarize_step(step))
            except Exception as exc:
                self.plan_steps_list.insert('end', f'Error loading plan: {exc}')

        self._refresh_gallery()

    def _create_plan_from_template(self):
        gpx_name = self.gpx_var.get()
        dest = flight_plan_path_for(gpx_name)
        if dest.exists():
            return
        dest.write_text(TEMPLATE_PLAN.read_text(encoding='utf-8'), encoding='utf-8')
        self._on_gpx_selected()

    def _open_plan_in_editor(self):
        plan_path = flight_plan_path_for(self.gpx_var.get())
        if not plan_path.exists():
            messagebox.showwarning('No flight plan', 'Create one from the template first.')
            return
        os.startfile(str(plan_path))

    # ── Gallery ──────────────────────────────────────────────────────────

    def _refresh_gallery(self):
        for child in self.gallery_inner.winfo_children():
            child.destroy()
        self.thumbnails.clear()

        gpx_name = self.gpx_var.get()
        preview_dir = RENDER_DIR / gpx_name / 'flight_preview'
        images = sorted(preview_dir.glob('*.png')) if preview_dir.exists() else []

        if not images:
            ttk.Label(self.gallery_inner, text='(no preview images yet)').grid(row=0, column=0, padx=4, pady=4)
            return

        for i, path in enumerate(images):
            try:
                img = tk.PhotoImage(file=str(path))
                factor = max(1, img.width() // THUMB_WIDTH)
                thumb = img.subsample(factor, factor)
            except Exception:
                continue
            self.thumbnails.append(thumb)

            cell = ttk.Frame(self.gallery_inner)
            cell.grid(row=i // GALLERY_COLUMNS, column=i % GALLERY_COLUMNS, padx=4, pady=4)
            ttk.Label(cell, image=thumb).pack()
            ttk.Label(cell, text=path.stem, wraplength=THUMB_WIDTH).pack()

    # ── Render lifecycle ─────────────────────────────────────────────────

    def _start_render(self):
        if self.proc is not None:
            return

        gpx_name = self.gpx_var.get()
        if not flight_plan_path_for(gpx_name).exists():
            messagebox.showerror('No flight plan',
                                  f'{flight_plan_path_for(gpx_name)} does not exist.\n'
                                  'Create one from the template first.')
            return

        try:
            start_frame = int(self.start_frame_var.get())
            if start_frame < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror('Invalid start frame', 'Start frame must be a non-negative integer.')
            return

        if not Path(BLENDER_EXE).exists():
            messagebox.showerror('Blender not found', f'{BLENDER_EXE} does not exist.')
            return

        preview = self.preview_var.get()

        env = os.environ.copy()
        env['GPX_NAME'] = gpx_name
        env['RENDER_PREVIEW'] = '1' if preview else '0'
        env['RENDER_ANIMATION'] = '0' if preview else '1'
        env['ANIMATION_START_FRAME'] = str(start_frame)

        gpx_path = GPX_DIR / f'{gpx_name}.gpx'
        if preview:
            try:
                plan = load_plan(gpx_name)
                self.progress_total = count_preview_images(plan)
            except Exception:
                self.progress_total = 1
            self.progress_mode = 'preview'
            self.progress_start_value = 0
        else:
            try:
                self.progress_total = estimate_total_frames(gpx_path)
            except Exception:
                self.progress_total = 1
            self.progress_mode = 'animation'
            self.progress_start_value = start_frame

        self.progress_current = self.progress_start_value
        self.first_progress_time = None
        self.first_progress_value = None
        self.progress_bar['value'] = 0
        self.progress_label.config(text=f'Starting ({self.progress_mode})...')

        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

        cmd = [BLENDER_EXE, '--background', '--python', str(ANIMATE_SCRIPT)]
        self._launch_process(cmd, env)

    def _start_generate_video(self):
        if self.proc is not None:
            return

        gpx_name = self.gpx_var.get()
        frames_dir = RENDER_DIR / gpx_name / 'animation_frames'
        frames = sorted(frames_dir.glob('frame_*.png')) if frames_dir.exists() else []
        if not frames:
            messagebox.showerror(
                'No frames found',
                f'No frame_*.png files found in {frames_dir}.\n\n'
                'Render the animation first. Note this step is only needed when the '
                'Blender build falls back to a PNG sequence instead of writing the MP4 directly.')
            return

        env = os.environ.copy()
        env['GPX_NAME'] = gpx_name

        self.progress_mode = 'video'
        self.progress_total = len(frames)
        self.progress_start_value = 0
        self.progress_current = 0
        self.first_progress_time = None
        self.first_progress_value = None
        self.progress_bar['value'] = 0
        self.progress_label.config(text='Starting (video)...')

        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

        cmd = [sys.executable, str(FRAMES_TO_MP4_SCRIPT)]
        self._launch_process(cmd, env)

    def _launch_process(self, cmd, env):
        # Both blender_animate.py and frames_to_mp4.py print non-ASCII characters
        # (e.g. '→'); with stdout piped (not a console), Python falls back to the
        # system codepage and crashes on them unless forced to UTF-8 here.
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        self.proc = subprocess.Popen(
            cmd, cwd=str(BASE_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding='utf-8', errors='replace', bufsize=1,
        )
        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

        self.render_btn.config(state='disabled')
        self.video_btn.config(state='disabled')
        self.cancel_btn.config(state='normal')

    def _cancel_render(self):
        if self.proc is not None:
            self.proc.terminate()

    def _read_process_output(self):
        for line in iter(self.proc.stdout.readline, ''):
            self.log_queue.put(line)
        self.log_queue.put(None)  # sentinel: process finished

    def _poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line is None:
                    self._on_process_finished()
                else:
                    self._handle_output_line(line)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _handle_output_line(self, line):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', line if line.endswith('\n') else line + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

        if self.progress_mode == 'animation':
            m = re.search(r'→\s*(\d+)\s*frames', line)
            if m:
                self.progress_total = int(m.group(1))
            m = re.search(r'Fra:(\d+)', line)
            if m:
                self._update_progress(int(m.group(1)))
        elif self.progress_mode == 'preview':
            if re.search(r'^\s*Preview:', line):
                self._update_progress(self.progress_current + 1)
        elif self.progress_mode == 'video':
            m = re.search(r'frame=\s*(\d+)', line)
            if m:
                self._update_progress(int(m.group(1)))

    def _update_progress(self, current_value):
        self.progress_current = current_value
        now = time.monotonic()
        if self.first_progress_time is None:
            self.first_progress_time = now
            self.first_progress_value = current_value

        span = max(1, self.progress_total - self.progress_start_value)
        done = max(0, current_value - self.progress_start_value)
        pct = min(100.0, 100.0 * done / span)
        self.progress_bar['value'] = pct

        elapsed = now - self.first_progress_time
        done_since_first = max(0, current_value - self.first_progress_value)
        eta = None
        if done_since_first > 0 and elapsed > 0:
            rate = done_since_first / elapsed
            remaining = max(0, self.progress_total - current_value)
            eta = remaining / rate if rate > 0 else None

        if self.progress_mode == 'animation':
            self.progress_label.config(
                text=f'Frame {current_value}/{self.progress_total} — ETA {format_eta(eta)}')
        elif self.progress_mode == 'preview':
            self.progress_label.config(
                text=f'Preview {current_value}/{self.progress_total} — ETA {format_eta(eta)}')
        else:
            self.progress_label.config(
                text=f'Encoding frame {current_value}/{self.progress_total} — ETA {format_eta(eta)}')

    def _on_process_finished(self):
        returncode = self.proc.wait() if self.proc else None
        self.proc = None
        self.render_btn.config(state='normal')
        self.video_btn.config(state='normal')
        self.cancel_btn.config(state='disabled')

        if returncode == 0:
            self.progress_bar['value'] = 100
            self.progress_label.config(text='Done.')
            if self.progress_mode == 'preview':
                self._refresh_gallery()
        else:
            self.progress_label.config(text=f'Stopped (exit code {returncode}).')


if __name__ == '__main__':
    App().mainloop()
