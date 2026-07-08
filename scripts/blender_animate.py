"""Blender animation entry point: DEM/contour/GPX pipeline + FlightPlan camera.

Run this script in Blender's Text Editor, or headless:
    blender --background --python scripts/blender_animate.py
"""

import os
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

import hashlib
import importlib.util

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as rio_transform
import gpxpy

try:
    from mathutils import Vector
except Exception:  # pragma: no cover - mathutils is not available outside Blender.
    class Vector(tuple):
        def __new__(cls, values):
            if isinstance(values, (list, tuple)):
                values = tuple(values)
            else:
                values = (values,)

            if len(values) == 2:
                values = values + (0.0,)
            elif len(values) != 3:
                raise ValueError(f'Unsupported vector values: {values}')

            return tuple.__new__(cls, tuple(float(v) for v in values))

        @property
        def x(self):
            return self[0]

        @property
        def y(self):
            return self[1]

        @property
        def z(self):
            return self[2]

        def __add__(self, other):
            if isinstance(other, Vector):
                return Vector((self.x + other.x, self.y + other.y, self.z + other.z))
            return Vector((self.x + other, self.y + other, self.z + other))

        def __radd__(self, other):
            return self.__add__(other)

        def __sub__(self, other):
            if isinstance(other, Vector):
                return Vector((self.x - other.x, self.y - other.y, self.z - other.z))
            return Vector((self.x - other, self.y - other, self.z - other))

        def __mul__(self, other):
            if isinstance(other, Vector):
                return Vector((self.x * other.x, self.y * other.y, self.z * other.z))
            return Vector((self.x * other, self.y * other, self.z * other))

        def __rmul__(self, other):
            return self.__mul__(other)

        def __truediv__(self, other):
            return Vector((self.x / other, self.y / other, self.z / other))

        def normalized(self):
            length = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
            if length == 0.0:
                return Vector((0.0, 0.0, 0.0))
            return self / length

        def to_track_quat(self, *args, **kwargs):
            return self

from gpx_trace import GPXTrace
from flight_plan import FlightPlan, Start, ForwardFollow, BackwardFollow, Rotate, CameraPose
from render_utils import configure_render, render_animation, set_bevel_linear

# =========================
# CONFIGURATION
# =========================

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(r"C:\Users\fabie\OneDrive\Documents\gpx-relief-tracer")
DEM_FOLDER      = BASE_DIR / r"contour_lines\isere\ign"
FLIGHT_PLANS_DIR = BASE_DIR / "flight_plans"
GPX_NAME = os.environ.get('GPX_NAME', r"Villard-de-Lans_Randonnée20260619215335")
GPX_PATH   = BASE_DIR / r"gpx" / f"{GPX_NAME}.gpx"

# ── Terrain ─────────────────────────────────────────────────────────────────
VERTICAL_EXAGGERATION = 1.5
DEM_MARGIN            = 1000.0  # metres of DEM loaded around the GPX bounding box

# ── Contour lines ────────────────────────────────────────────────────────────
CONTOUR_INTERVAL  = 20      # metres between contour levels
CONTOUR_THICKNESS = 0.0003  # fraction of terrain extent

# ── GPX track ────────────────────────────────────────────────────────────────
GPX_THICKNESS = 0.003  # fraction of terrain extent

# ── Materials ────────────────────────────────────────────────────────────────
TERRAIN_COLOR             = (0.0, 0.0, 0.0, 1.0)  # RGBA
CONTOUR_COLOR             = (1.0, 1.0, 1.0, 1.0)  # RGBA
CONTOUR_EMISSION_STRENGTH = 3.0
GPX_COLOR                 = (1.0, 0.0, 0.0, 1.0)  # RGBA
GPX_EMISSION_STRENGTH     = 20.0

# ── Render ───────────────────────────────────────────────────────────────────
QUICK_RENDER         = True   # True → fast preview; False → full quality
RENDER_SAMPLES_QUICK = 16
RENDER_SAMPLES_FULL  = 64
RENDER_RES_QUICK     = (854, 480)
RENDER_RES_FULL      = (1920, 1080)

# =========================
# LOAD GPX
# =========================

def load_gpx(path):
    with open(path, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)

    lon = []
    lat = []
    elev = []
    timestamps = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                lon.append(point.longitude)
                lat.append(point.latitude)
                elev.append(point.elevation if point.elevation is not None else np.nan)
                timestamps.append(point.time)  # datetime or None

    return (np.array(lon, dtype=np.float64), np.array(lat, dtype=np.float64),
            np.array(elev, dtype=np.float32), timestamps)


def infer_dem_crs(dem_crs, dem_path):
    if dem_crs is not None:
        return dem_crs

    path_text = str(dem_path).upper()
    if 'LAMB93' in path_text or 'LAMBERT' in path_text or 'IGN69' in path_text:
        return 'EPSG:2154'
    return None


def gpx_to_dem_coords(lon, lat, dem_crs, dem_path=None):
    target_crs = infer_dem_crs(dem_crs, dem_path)
    if target_crs is None:
        return lon, lat

    crs_text = str(target_crs)
    if crs_text.lower().startswith('epsg:4326'):
        return lon, lat

    x, y = rio_transform('EPSG:4326', target_crs, lon.tolist(), lat.tolist())
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def bounds_from_points(x, y, margin=1000.0):
    min_x = float(np.nanmin(x))
    max_x = float(np.nanmax(x))
    min_y = float(np.nanmin(y))
    max_y = float(np.nanmax(y))
    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def discover_dem_candidates(base_dir=None):
    search_root = Path(base_dir or DEM_FOLDER)
    return sorted(path for path in search_root.rglob('*.asc') if path.is_file())


def select_dem_tiles_for_gpx(candidates, gpx_x, gpx_y):
    print(f'  Selecting DEM tiles from {len(candidates)} candidates...')
    candidate_paths = [Path(path) for path in candidates]
    if not candidate_paths:
        return []

    selected = []
    if len(gpx_x) != len(gpx_y):
        return candidate_paths[:1]

    for path in candidate_paths:
        try:
            with rasterio.open(path) as ds:
                bounds = ds.bounds
                for i in range(1, len(gpx_x)):
                    x0 = float(gpx_x[i - 1])
                    y0 = float(gpx_y[i - 1])
                    x1 = float(gpx_x[i])
                    y1 = float(gpx_y[i])
                    if np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1):
                        continue

                    segment_min_x = min(x0, x1)
                    segment_max_x = max(x0, x1)
                    segment_min_y = min(y0, y1)
                    segment_max_y = max(y0, y1)
                    overlap_x = max(0.0, min(segment_max_x, bounds.right) - max(segment_min_x, bounds.left))
                    overlap_y = max(0.0, min(segment_max_y, bounds.top) - max(segment_min_y, bounds.bottom))
                    if overlap_x > 0.0 and overlap_y > 0.0:
                        selected.append(path)
                        break
        except Exception as exc:
            print(f'  Skipping DEM candidate {path}: {exc}')

    if not selected:
        return candidate_paths[:1]
    return list(dict.fromkeys(selected))


def merge_dem_tiles(tile_paths):
    if not tile_paths:
        return None, None, None

    loaded_tiles = []
    for path in tile_paths:
        with rasterio.open(path) as ds:
            loaded_tiles.append((Path(path), ds.read(1).astype(np.float32), ds.transform, ds.crs, ds.bounds))

    if not loaded_tiles:
        return None, None, None

    first_path, first_elev, first_transform, first_crs, first_bounds = loaded_tiles[0]
    left = min(bounds.left for _, _, _, _, bounds in loaded_tiles)
    right = max(bounds.right for _, _, _, _, bounds in loaded_tiles)
    bottom = min(bounds.bottom for _, _, _, _, bounds in loaded_tiles)
    top = max(bounds.top for _, _, _, _, bounds in loaded_tiles)

    cellsize = abs(first_transform.a)
    width = max(1, int(np.ceil((right - left) / cellsize)))
    height = max(1, int(np.ceil((top - bottom) / cellsize)))

    mosaic = np.full((height, width), np.nan, dtype=np.float32)
    global_transform = Affine(cellsize, 0.0, left, 0.0, -cellsize, top)

    for _, elev, _, _, bounds in loaded_tiles:
        col0, row0 = ~global_transform * (bounds.left, bounds.top)
        col0 = int(np.floor(col0))
        row0 = int(np.floor(row0))

        dst_col0 = max(0, col0)
        dst_row0 = max(0, row0)
        src_col0 = max(0, -col0)
        src_row0 = max(0, -row0)

        dst_col1 = min(width, col0 + elev.shape[1])
        dst_row1 = min(height, row0 + elev.shape[0])
        src_col1 = src_col0 + (dst_col1 - dst_col0)
        src_row1 = src_row0 + (dst_row1 - dst_row0)

        if dst_col1 <= dst_col0 or dst_row1 <= dst_row0:
            continue

        mosaic[dst_row0:dst_row1, dst_col0:dst_col1] = elev[src_row0:src_row1, src_col0:src_col1]

    return mosaic, global_transform, first_crs


def crop_dem_to_bounds(elev, transform, min_x, max_x, min_y, max_y):
    inv = ~transform
    col0, row0 = inv * (min_x, max_y)
    col1, row1 = inv * (max_x, min_y)

    col0 = int(np.floor(col0))
    row0 = int(np.floor(row0))
    col1 = int(np.ceil(col1))
    row1 = int(np.ceil(row1))

    col0 = max(0, col0)
    row0 = max(0, row0)
    col1 = min(elev.shape[1], col1)
    row1 = min(elev.shape[0], row1)

    if col1 <= col0 or row1 <= row0:
        return elev, transform

    cropped = elev[row0:row1, col0:col1]
    new_transform = transform * Affine.translation(col0, row0)
    return cropped, new_transform


def contour_cache_path(dem_paths, gpx_path, interval, bounds):
    gpx_path = Path(gpx_path)

    def stat_fields(path):
        try:
            st = path.stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return 0, 0

    if isinstance(dem_paths, (list, tuple, set)):
        dem_paths = [Path(path) for path in dem_paths]
        dem_mtime = sum(st.st_mtime_ns for st in (path.stat() if path.exists() else Path('.').stat() for path in dem_paths))
        dem_size = sum(st.st_size for st in (path.stat() if path.exists() else Path('.').stat() for path in dem_paths))
    else:
        dem_path = Path(dem_paths)
        dem_mtime, dem_size = stat_fields(dem_path)

    gpx_mtime, gpx_size = stat_fields(gpx_path)

    key = (
        f"{dem_paths}|{gpx_path}|{interval}|"
        f"{bounds[0]:.3f}|{bounds[1]:.3f}|{bounds[2]:.3f}|{bounds[3]:.3f}|"
        f"{dem_mtime}|{dem_size}|{gpx_mtime}|{gpx_size}"
    )
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]
    return BASE_DIR / Path(f"cache") / f"contours_cache_{digest}.npz"


def save_contour_cache(cache_path, segments):
    arr = np.array(segments, dtype=np.float32)
    np.savez_compressed(cache_path, segments=arr)


def load_contour_cache(cache_path):
    with np.load(cache_path) as data:
        arr = data['segments']
        return [ (tuple(row[0]), tuple(row[1])) for row in arr ]


def sample_dem_elevation(x, y, dem_elev, transform):
    col, row = ~transform * (x, y)
    col = int(round(col))
    row = int(round(row))
    if 0 <= row < dem_elev.shape[0] and 0 <= col < dem_elev.shape[1]:
        value = dem_elev[row, col]
        return float(value) if not np.isnan(value) else None
    return None


def clean_scene():
    if bpy is None:
        return

    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA' or obj.name.startswith('terrain') or obj.name.startswith('contours') or obj.name.startswith('GPX'):
            bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        if mesh.name.startswith('terrain') or mesh.name.startswith('contours'):
            bpy.data.meshes.remove(mesh, do_unlink=True)

    for curve in list(bpy.data.curves):
        if curve.name.startswith('gpx_curve') or curve.name.startswith('contours'):
            bpy.data.curves.remove(curve, do_unlink=True)

    for camera in list(bpy.data.cameras):
        bpy.data.cameras.remove(camera, do_unlink=True)

    if bpy.context.scene.camera is not None:
        bpy.context.scene.camera = None


# =========================
# CREATE TERRAIN MESH
# =========================

def create_terrain(elev, transform):
    if bpy is None:
        return None

    h, w = elev.shape
    verts = []
    faces = []

    for y in range(h):
        for x in range(w):
            z = float(elev[y, x]) * VERTICAL_EXAGGERATION
            wx, wy = transform * (x, y)
            verts.append((wx, wy, z))

    def idx(x, y):
        return y * w + x

    for y in range(h - 1):
        for x in range(w - 1):
            faces.append([
                idx(x, y),
                idx(x + 1, y),
                idx(x + 1, y + 1),
                idx(x, y + 1),
            ])

    mesh = bpy.data.meshes.new('terrain')
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new('terrain', mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# =========================
# CONTOURS
# =========================

def sample_edge(p0, p1, z0, z1, level):
    if z1 == z0:
        t = 0.5
    else:
        t = (level - z0) / (z1 - z0)
    return (
        p0[0] + (p1[0] - p0[0]) * t,
        p0[1] + (p1[1] - p0[1]) * t,
        level * VERTICAL_EXAGGERATION,
    )


def generate_contour_segments(elev, transform, interval=10):
    segments = []
    h, w = elev.shape

    min_z = float(np.nanmin(elev))
    max_z = float(np.nanmax(elev))
    start_level = np.ceil(min_z / interval) * interval
    levels = np.arange(start_level, max_z + interval, interval, dtype=np.float32)

    total_cells = len(levels) * (h - 1) * (w - 1)
    if total_cells <= 0:
        return segments

    report_step = max(1, total_cells // 50)
    processed = 0
    print(f'  Generating contour segments for {len(levels)} levels ({total_cells} cells)')

    for level in levels:
        for row in range(h - 1):
            for col in range(w - 1):
                z0 = float(elev[row, col])
                z1 = float(elev[row, col + 1])
                z2 = float(elev[row + 1, col + 1])
                z3 = float(elev[row + 1, col])

                if np.isnan(z0) or np.isnan(z1) or np.isnan(z2) or np.isnan(z3):
                    processed += 1
                    if processed % report_step == 0 or processed == total_cells:
                        pct = int(processed * 100 / total_cells)
                        print(f'    contour progress: {pct}% ({processed}/{total_cells})', end='\r', flush=True)
                    continue

                case = 0
                case |= 1 if z0 >= level else 0
                case |= 2 if z1 >= level else 0
                case |= 4 if z2 >= level else 0
                case |= 8 if z3 >= level else 0

                if case == 0 or case == 15:
                    processed += 1
                    if processed % report_step == 0 or processed == total_cells:
                        pct = int(processed * 100 / total_cells)
                        print(f'    contour progress: {pct}% ({processed}/{total_cells})', end='\r', flush=True)
                    continue

                p0 = transform * (col, row)
                p1 = transform * (col + 1, row)
                p2 = transform * (col + 1, row + 1)
                p3 = transform * (col, row + 1)

                bottom = sample_edge(p0, p1, z0, z1, level)
                right = sample_edge(p1, p2, z1, z2, level)
                top = sample_edge(p3, p2, z3, z2, level)
                left = sample_edge(p0, p3, z0, z3, level)

                if case in (1, 14):
                    segments.append((left, bottom))
                elif case in (2, 13):
                    segments.append((bottom, right))
                elif case in (3, 12):
                    segments.append((left, right))
                elif case in (4, 11):
                    segments.append((right, top))
                elif case == 5:
                    segments.append((bottom, top))
                    segments.append((left, right))
                elif case == 6:
                    segments.append((bottom, top))
                elif case == 7:
                    segments.append((left, top))
                elif case == 8:
                    segments.append((left, top))
                elif case == 9:
                    segments.append((bottom, top))
                elif case == 10:
                    segments.append((bottom, right))
                    segments.append((left, top))

                processed += 1
                if processed % report_step == 0 or processed == total_cells:
                    pct = int(processed * 100 / total_cells)
                    print(f'    contour progress: {pct}% ({processed}/{total_cells})', end='\r', flush=True)

    print()
    return segments


def create_contour_object(segments):
    if bpy is None:
        return None

    curve = bpy.data.curves.new('contours', type='CURVE')
    curve.dimensions = '3D'

    terrain = bpy.data.objects.get('terrain')
    extent = 1.0
    if terrain is not None:
        try:
            bbox_world = [terrain.matrix_world @ Vector(corner) for corner in terrain.bound_box]
            xs = [v.x for v in bbox_world]
            ys = [v.y for v in bbox_world]
            zs = [v.z for v in bbox_world]
            extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
        except Exception:
            extent = 1.0

    curve.bevel_depth = extent * CONTOUR_THICKNESS
    curve.resolution_u = 1

    for start, end in segments:
        spline = curve.splines.new('POLY')
        spline.points.add(1)  # starts with 1 point; add 1 more → 2 total
        spline.points[0].co = (*start, 1.0)
        spline.points[1].co = (*end, 1.0)

    obj = bpy.data.objects.new('contours', curve)
    bpy.context.collection.objects.link(obj)
    return obj

# =========================
# PROJECT GPX ON TERRAIN
# =========================

def project_gpx(lon, lat, elevations, dem_elev, transform, dem_obj, ray_height):
    if bpy is None:
        return []

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    projected = []

    for x, y, elev in zip(lon, lat, elevations):
        origin = Vector((x, y, ray_height))
        direction = Vector((0.0, 0.0, -1.0))
        result, location, normal, face_index, hit_obj, matrix = scene.ray_cast(depsgraph, origin, direction)
        if result and hit_obj == dem_obj:
            projected.append(location)
            continue

        dem_z = sample_dem_elevation(x, y, dem_elev, transform)
        if dem_z is not None:
            projected.append(Vector((x, y, dem_z * VERTICAL_EXAGGERATION)))
            continue

        if not np.isnan(elev):
            projected.append(Vector((x, y, elev * VERTICAL_EXAGGERATION)))

    return projected


def create_gpx_curve(points):
    if bpy is None:
        return None

    if len(points) < 2:
        return None

    # Debug info about the GPX curve placement
    try:
        coords = np.array([(p.x, p.y, p.z) for p in points], dtype=np.float64)
        if len(coords) > 0:
            print(f'  GPX curve start: {coords[0]}')
            print(f'  GPX curve end: {coords[-1]}')
            print(f'  GPX curve points: {len(coords)}')
            if len(coords) > 1:
                diffs = coords[-1] - coords[0]
                dist = float(np.linalg.norm(diffs))
                print(f'  GPX curve length estimate: {dist:.3f}')
    except Exception as e:
        print(f'  GPX curve debug logging failed: {e}')

    curve = bpy.data.curves.new('gpx_curve', type='CURVE')
    curve.dimensions = '3D'
    spline = curve.splines.new('POLY')
    spline.points.add(len(points) - 1)

    # determine a reasonable bevel thickness and Z offset based on terrain size
    terrain = bpy.data.objects.get('terrain')
    extent = 1.0
    if terrain is not None:
        try:
            bbox_world = [terrain.matrix_world @ Vector(corner) for corner in terrain.bound_box]
            xs = [v.x for v in bbox_world]
            ys = [v.y for v in bbox_world]
            zs = [v.z for v in bbox_world]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            dz = max(zs) - min(zs)
            extent = max(dx, dy, dz, 1.0)
        except Exception:
            extent = 1.0

    bevel = extent * GPX_THICKNESS
    z_offset = max(1.0, extent * 0.001)

    for i, point in enumerate(points):
        px = point.x
        py = point.y
        pz = point.z + z_offset
        spline.points[i].co = (px, py, pz, 1.0)

    curve.bevel_depth = bevel
    curve.resolution_u = 2

    obj = bpy.data.objects.new('GPX', curve)
    bpy.context.collection.objects.link(obj)

    return obj


def setup_materials():
    if bpy is None:
        return {}

    def make_emission(name, color, strength=5.0):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs[0].default_value = color
        emission.inputs[1].default_value = strength
        links.new(emission.outputs[0], output.inputs[0])
        return mat

    # terrain should be a black material
    terrain_mat = bpy.data.materials.new(name='terrain_mat')
    terrain_mat.use_nodes = True
    nodes = terrain_mat.node_tree.nodes
    links = terrain_mat.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = TERRAIN_COLOR
    principled.inputs['Roughness'].default_value = 1.0
    links.new(principled.outputs[0], output.inputs[0])

    return {
        'terrain': terrain_mat,
        'contours': make_emission('contour_mat', CONTOUR_COLOR, CONTOUR_EMISSION_STRENGTH),
        'gpx': make_emission('gpx_mat', GPX_COLOR, GPX_EMISSION_STRENGTH),
    }


# =============================================================================
# FLIGHT PLAN — per-GPX camera plan, loaded from flight_plans/<GPX_NAME>.py
# =============================================================================

ANIMATION_FPS   = 24
ANIMATION_SPEED = 600   # real-time multiplier (600 → 3 h hike becomes ~34 s video)


def load_flight_plan(gpx_name: str) -> FlightPlan:
    """Import flight_plans/<gpx_name>.py and return its PLAN.

    Keeps route-specific camera tuning out of this shared pipeline script —
    add a new file there for each new GPX instead of editing this one.
    """
    plan_path = FLIGHT_PLANS_DIR / f'{gpx_name}.py'
    if not plan_path.exists():
        raise FileNotFoundError(
            f'No flight plan found for {gpx_name!r}. '
            f'Create {plan_path} (copy flight_plans/_template.py as a starting point).'
        )
    spec = importlib.util.spec_from_file_location(f'flight_plans.{gpx_name}', plan_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PLAN


PLAN = load_flight_plan(GPX_NAME)

# Output controls
# All three may be overridden by environment variables (set by scripts/gui.py)
# so the shared pipeline script itself never needs editing for a headless run.
RENDER_PREVIEW        = os.environ.get('RENDER_PREVIEW', '0') == '1'    # True → render preview images only (fast, for tuning)
RENDER_ANIMATION      = os.environ.get('RENDER_ANIMATION', '1') == '1'  # True → auto-render all frames (slow)
ANIMATION_START_FRAME = int(os.environ.get('ANIMATION_START_FRAME', '0'))  # Resume from this frame (0 = start from beginning)

PREVIEW_DIR  = BASE_DIR / 'render' / GPX_NAME / 'flight_preview'
VIDEO_OUTPUT = BASE_DIR / 'render' / GPX_NAME / 'animation.mp4'

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
    configure_render(scene, QUICK_RENDER, RENDER_SAMPLES_QUICK, RENDER_SAMPLES_FULL,
                      RENDER_RES_QUICK, RENDER_RES_FULL)

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
    start_frame = max(0, min(ANIMATION_START_FRAME, total_frames))

    scene.render.fps  = ANIMATION_FPS
    scene.frame_start = start_frame
    scene.frame_end   = total_frames

    cam = _add_camera()
    plan._reset()

    print(f'  Setting keyframes: frames {start_frame}–{total_frames}...')
    for frame in range(total_frames + 1):
        t = frame / total_frames

        # Always advance EMA so the camera state is correct from start_frame onward
        pose = plan.camera_pose(t, gpx_trace)

        if frame < start_frame:
            if frame % 100 == 0:
                print(f'    warming EMA: {frame}/{start_frame - 1}', end='\r', flush=True)
            continue

        # GPX trace reveal
        gpx_obj.data.bevel_factor_end = gpx_trace.bevel_factor_at(t)
        gpx_obj.data.keyframe_insert(data_path='bevel_factor_end', frame=frame)

        # Camera
        _set_camera_pose(cam, pose)
        cam.keyframe_insert(data_path='location',       frame=frame)
        cam.keyframe_insert(data_path='rotation_euler', frame=frame)

        if frame % 100 == 0:
            print(f'    frame {frame}/{total_frames}', end='\r', flush=True)

    print()
    set_bevel_linear(gpx_obj)

    if RENDER_ANIMATION:
        configure_render(scene, QUICK_RENDER, RENDER_SAMPLES_QUICK, RENDER_SAMPLES_FULL,
                          RENDER_RES_QUICK, RENDER_RES_FULL)
        render_animation(scene, VIDEO_OUTPUT, ANIMATION_FPS, total_frames)
    else:
        print(f'  Keyframes set. Press Ctrl+F12 in Blender to render animation.')
        print(f'  Configure output in Properties → Output before rendering.')


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
