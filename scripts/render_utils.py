"""Render-setup helpers used by blender_animate.py.

Configures the Cycles engine, renders an animation to FFMPEG with a
PNG-sequence fallback, and patches bevel_factor_end keyframes to LINEAR
interpolation across Blender's pre-4.4 and 4.4+ (layered action) FCurve APIs.
"""

try:
    import bpy
except Exception:  # pragma: no cover - Blender is not available in the test environment.
    bpy = None


def configure_render(scene, quick_render, samples_quick, samples_full, res_quick, res_full):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.device = 'CPU'
        scene.cycles.samples = samples_quick if quick_render else samples_full
    except Exception:
        pass
    res_x, res_y = res_quick if quick_render else res_full
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y


def render_animation(scene, video_output, fps, total_frames):
    """Render all keyframed frames to video_output via FFMPEG.

    Falls back to a PNG sequence (in a `<stem>_frames` sibling directory) when
    the running Blender build has no FFMPEG muxer, printing the ffmpeg command
    needed to assemble it afterwards.
    """
    ffmpeg_ok = False
    frames_dir = None
    try:
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        try:
            scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
        except Exception:
            pass
        scene.render.filepath = str(video_output)
        ffmpeg_ok = True
    except (TypeError, AttributeError):
        frames_dir = video_output.parent / (video_output.stem + '_frames')
        frames_dir.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = str(frames_dir / 'frame_')
        print('  FFMPEG format not available in this Blender build.')
        print(f'  Rendering PNG frames to: {frames_dir}')
        print('  Combine afterwards with:')
        print(f'    ffmpeg -r {fps} -i "{frames_dir}\\frame_%04d.png" -c:v libx264 -pix_fmt yuv420p "{video_output}"')

    out = video_output if ffmpeg_ok else frames_dir
    print(f'  Rendering {total_frames} frames → {out}')
    bpy.ops.render.render(animation=True)
    print('  Done.')


def set_bevel_linear(gpx_obj):
    """Set bevel_factor_end keyframes to LINEAR interpolation (Blender < 4.4 and 4.4+ layered actions)."""
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
        print(f'  Warning: could not set bevel interpolation to LINEAR: {e}')
