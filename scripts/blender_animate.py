"""Blender animation entry point using the FlightPlan camera system.

Run this script in Blender's Text Editor instead of blender_tracer.py
when you want a custom multi-segment flight plan.

It reuses all terrain/GPX/material pipeline functions from blender_tracer.py
and adds only the flight-plan camera logic.
"""

import sys
from pathlib import Path

# Ensure sibling scripts are importable when running inside Blender's Text Editor
_SCRIPTS = Path(r"C:\Users\fabie\OneDrive\Documents\gpx-relief-tracer\scripts")
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

try:
    import bpy
except ImportError:
    bpy = None

import numpy as np

from gpx_trace import GPXTrace
from flight_plan import FlightPlan, Start, ForwardFollow, BackwardFollow, Rotate, CameraPose

# Import the full terrain/GPX pipeline from blender_tracer without running main()
from blender_tracer import (
    BASE_DIR, DEM_FOLDER, GPX_PATH, DEM_MARGIN, CONTOUR_INTERVAL,
    VERTICAL_EXAGGERATION, QUICK_RENDER,
    RENDER_SAMPLES_QUICK, RENDER_SAMPLES_FULL, RENDER_RES_QUICK, RENDER_RES_FULL,
    load_gpx, discover_dem_candidates, gpx_to_dem_coords,
    select_dem_tiles_for_gpx, bounds_from_points,
    merge_dem_tiles, crop_dem_to_bounds,
    contour_cache_path, load_contour_cache, save_contour_cache,
    generate_contour_segments, create_terrain, create_contour_object,
    project_gpx, create_gpx_curve, setup_materials, clean_scene,
)

# =============================================================================
# FLIGHT PLAN — edit this section to customise the camera
# =============================================================================

ANIMATION_FPS   = 24
ANIMATION_SPEED = 600   # real-time multiplier (600 → 3 h hike becomes ~34 s video)

PLAN = FlightPlan(
    steps=[
        # Start defines where the camera is at t=0.
        # Set end_t=0.0 so it is instantaneous (no preview image generated).
        Start(end_t=0.0, azimuth=140.0, elevation=25.0, distance=5000.0),

        # First half: trail behind the trace head
        ForwardFollow(end_t=0.5, distance=3000.0, height=500.0, look_ahead=200.0),

        # Sweep around to a different vantage point
        Rotate(end_t=0.62, end_azimuth=235.0, end_elevation=30.0, distance=3000.0),

        # Second half: lead ahead of the trace head, looking back
        BackwardFollow(end_t=1.0, distance=3000.0, height=500.0),
    ],
    smoothing=0.005,   # EMA factor; lower = floatier, higher = snappier
)

# Output controls
RENDER_PREVIEW   = True    # True → render preview images only (fast, for tuning)
RENDER_ANIMATION = False   # True → auto-render all frames (slow)

PREVIEW_DIR  = BASE_DIR / 'render' / 'flight_preview'
VIDEO_OUTPUT = BASE_DIR / 'render' / 'animation.mp4'

# =============================================================================


def _set_camera_pose(cam, pose: CameraPose):
    """Position and orient a Blender camera object from a CameraPose."""
    from mathutils import Vector
    pos = Vector(pose.position)
    tgt = Vector(pose.target)
    cam.location = pos
    direction = (tgt - pos).normalized()
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def _add_camera() -> object:
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.data.clip_start = 1.0
    cam.data.clip_end   = 50_000.0
    bpy.context.scene.camera = cam
    return cam


def _configure_render(scene):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.device  = 'CPU'
        scene.cycles.samples = RENDER_SAMPLES_QUICK if QUICK_RENDER else RENDER_SAMPLES_FULL
    except Exception:
        pass
    res_x, res_y = RENDER_RES_QUICK if QUICK_RENDER else RENDER_RES_FULL
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y


def _set_bevel_linear(gpx_obj):
    """Try to set bevel_factor_end keyframes to LINEAR interpolation (Blender 4.x and 5.x)."""
    try:
        action = gpx_obj.data.animation_data and gpx_obj.data.animation_data.action
        if not action:
            return
        fcurves = None
        try:
            fcurves = action.fcurves
        except AttributeError:
            try:
                fcurves = action.layers[0].strips[0].channelbags[0].fcurves
            except (AttributeError, IndexError):
                pass
        if fcurves:
            for fc in fcurves:
                if fc.data_path == 'bevel_factor_end':
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'
    except Exception as e:
        print(f'  Warning: could not set bevel interpolation: {e}')


# =============================================================================
# Render modes
# =============================================================================

def render_preview(plan: FlightPlan, gpx_trace: GPXTrace, gpx_obj):
    """Render 3 PNG images per non-trivial flight plan step (start, mid, end)."""
    if bpy is None:
        return

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    cam = _add_camera()
    _configure_render(scene)

    previews = plan.preview_poses(gpx_trace)  # list of (t, step, label, pose)
    step_counts = {}

    for t, step, label, pose in previews:
        step_name = type(step).__name__
        step_num  = step_counts.setdefault(id(step), len(step_counts))

        gpx_obj.data.bevel_factor_end = gpx_trace.bevel_factor_at(t)
        _set_camera_pose(cam, pose)

        filename = f'step{step_num:02d}_{step_name}_{label}.png'
        scene.render.filepath = str(PREVIEW_DIR / filename)
        print(f'  Preview: {filename}  (t={t:.3f})')
        bpy.ops.render.render(write_still=True)

    print(f'  {len(previews)} preview images saved to {PREVIEW_DIR}')


def apply_flight_plan(plan: FlightPlan, gpx_trace: GPXTrace, gpx_obj, total_frames: int):
    """Set Blender keyframes for the GPX reveal and camera for every frame."""
    if bpy is None:
        return

    scene = bpy.context.scene
    scene.render.fps  = ANIMATION_FPS
    scene.frame_start = 0
    scene.frame_end   = total_frames

    cam = _add_camera()
    plan._reset()

    print(f'  Setting keyframes: {total_frames + 1} frames...')
    for frame in range(total_frames + 1):
        t = frame / total_frames

        # GPX trace reveal
        gpx_obj.data.bevel_factor_end = gpx_trace.bevel_factor_at(t)
        gpx_obj.data.keyframe_insert(data_path='bevel_factor_end', frame=frame)

        # Camera
        pose = plan.camera_pose(t, gpx_trace)
        _set_camera_pose(cam, pose)
        cam.keyframe_insert(data_path='location',       frame=frame)
        cam.keyframe_insert(data_path='rotation_euler', frame=frame)

        if frame % 100 == 0:
            print(f'    frame {frame}/{total_frames}', end='\r', flush=True)

    print()
    _set_bevel_linear(gpx_obj)

    if RENDER_ANIMATION:
        _render_video(scene, total_frames)
    else:
        print(f'  Keyframes set. Press Ctrl+F12 in Blender to render animation.')
        print(f'  Configure output in Properties → Output before rendering.')


def _render_video(scene, total_frames: int):
    _configure_render(scene)
    try:
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec  = 'H264'
        try:
            scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
        except Exception:
            pass
        scene.render.filepath = str(VIDEO_OUTPUT)
    except (TypeError, AttributeError):
        frames_dir = VIDEO_OUTPUT.parent / (VIDEO_OUTPUT.stem + '_frames')
        frames_dir.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = str(frames_dir / 'frame_')
        print(f'  FFMPEG unavailable; rendering PNG frames to {frames_dir}')
        print(f'  Combine with:')
        print(f'    ffmpeg -r {ANIMATION_FPS} -i "{frames_dir}\\frame_%04d.png"'
              f' -c:v libx264 -pix_fmt yuv420p "{VIDEO_OUTPUT}"')

    print(f'  Rendering {total_frames} frames → {scene.render.filepath}')
    bpy.ops.render.render(animation=True)
    print('  Done.')


# =============================================================================
# Pipeline
# =============================================================================

def _video_duration(timestamps) -> float:
    valid = [ts for ts in timestamps if ts is not None]
    if len(valid) >= 2:
        return (valid[-1] - valid[0]).total_seconds() / ANIMATION_SPEED
    return 60.0


def main():
    print('=== blender_animate.py ===')

    print('1/7  Cleaning scene...')
    clean_scene()

    print('2/7  Loading GPX...')
    lon, lat, gpx_elev, timestamps = load_gpx(GPX_PATH)
    ts_count = sum(1 for t in timestamps if t is not None)
    print(f'     {len(lon)} points, {ts_count} with timestamps')

    print('3/7  Selecting DEM tiles...')
    candidates = discover_dem_candidates(DEM_FOLDER)
    sample     = candidates[0] if candidates else DEM_FOLDER
    gpx_x, gpx_y = gpx_to_dem_coords(lon, lat, None, sample)
    selected = select_dem_tiles_for_gpx(candidates, gpx_x, gpx_y)
    min_x, max_x, min_y, max_y = bounds_from_points(gpx_x, gpx_y, margin=DEM_MARGIN)
    print(f'     {len(selected)} tiles selected')

    print('4/7  Loading and merging DEM...')
    elev_grid, transform, dem_crs = merge_dem_tiles(selected)
    elev, transform = crop_dem_to_bounds(elev_grid, transform, min_x, max_x, min_y, max_y)

    print('5/7  Terrain mesh...')
    terrain = create_terrain(elev, transform)

    print('6/7  Contour lines...')
    cache_path = contour_cache_path(selected, GPX_PATH, CONTOUR_INTERVAL,
                                     (min_x, max_x, min_y, max_y))
    if cache_path.exists():
        segments = load_contour_cache(cache_path)
        print(f'     {len(segments)} segments from cache')
    else:
        segments = generate_contour_segments(elev, transform, CONTOUR_INTERVAL)
        save_contour_cache(cache_path, segments)
    contour_obj = create_contour_object(segments)

    print('7/7  GPX projection...')
    ray_height = float(np.nanmax(elev)) * VERTICAL_EXAGGERATION + 1000.0
    projected  = project_gpx(gpx_x, gpx_y, gpx_elev, elev, transform, terrain, ray_height)
    gpx_obj    = create_gpx_curve(projected)
    print(f'     {len(projected)} projected points')

    mats = setup_materials()
    terrain.active_material    = mats['terrain']
    contour_obj.active_material = mats['contours']
    if gpx_obj:
        gpx_obj.active_material = mats['gpx']

    gpx_trace     = GPXTrace(projected, timestamps)
    video_seconds = _video_duration(timestamps)
    total_frames  = max(1, int(video_seconds * ANIMATION_FPS))
    print(f'     Video: {video_seconds:.1f}s → {total_frames} frames at {ANIMATION_FPS} fps')

    if RENDER_PREVIEW:
        print('Flight plan preview...')
        render_preview(PLAN, gpx_trace, gpx_obj)
    else:
        print('Applying flight plan...')
        apply_flight_plan(PLAN, gpx_trace, gpx_obj, total_frames)

    print('DONE.')


if __name__ == '__main__':
    main()