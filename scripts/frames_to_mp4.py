"""Convert the PNG frame sequence produced by Blender into an MP4 video.

Usage:
    python frames_to_mp4.py

Requires ffmpeg to be installed and available on PATH.
Download from https://ffmpeg.org/download.html
"""

from pathlib import Path
import subprocess
import sys

from scripts.blender_animate import GPX_NAME

# =============================================================================
# CONFIGURATION — adjust these to match your render settings
# =============================================================================

BASE_DIR    = Path(r"C:\Users\fabie\OneDrive\Documents\gpx-relief-tracer")
FRAMES_DIR  = BASE_DIR / 'render' / GPX_NAME / 'animation_frames'
OUTPUT_FILE = BASE_DIR / 'render' / GPX_NAME / 'render.mp4'

FPS         = 24      # must match ANIMATION_FPS in blender_animate.py
CRF         = 18      # quality: 0 (lossless) → 51 (worst); 18 is visually near-lossless
PRESET      = 'slow'  # encoding speed/compression trade-off: ultrafast … slow … veryslow

# =============================================================================

def main():
    # Discover frames and validate
    frames = sorted(FRAMES_DIR.glob('frame_*.png'))
    if not frames:
        print(f'ERROR: No frame_*.png files found in {FRAMES_DIR}')
        sys.exit(1)

    print(f'Found {len(frames)} frames in {FRAMES_DIR}')
    print(f'Output → {OUTPUT_FILE}')

    # Infer the zero-padding width from the first filename (e.g. frame_0000 → 4)
    first_stem = frames[0].stem          # e.g. 'frame_0000'
    number_part = first_stem.split('_', 1)[1]
    pad = len(number_part)               # typically 4

    input_pattern = str(FRAMES_DIR / f'frame_%0{pad}d.png')

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-y',                            # overwrite without asking
        '-framerate', str(FPS),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-crf', str(CRF),
        '-preset', PRESET,
        '-pix_fmt', 'yuv420p',           # required for broad player compatibility
        str(OUTPUT_FILE),
    ]

    print('Running:', ' '.join(f'"{a}"' if ' ' in a else a for a in cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print('ERROR: ffmpeg not found on PATH.')
        print('Install it from https://ffmpeg.org/download.html and make sure')
        print('the ffmpeg binary directory is added to your system PATH.')
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f'ERROR: ffmpeg exited with code {e.returncode}')
        sys.exit(e.returncode)

    size_mb = OUTPUT_FILE.stat().st_size / 1_048_576
    print(f'\nDone. {OUTPUT_FILE} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
