import bpy
import numpy as np
import rasterio
import gpxpy
from math import radians
from mathutils import Vector

# =========================
# CONFIGURATION
# =========================

DEM_PATH = "/path/to/IGN_RGE_ALTI.tif"
GPX_PATH = "/path/to/track.gpx"

VERTICAL_EXAGGERATION = 1.5
CONTOUR_INTERVAL = 10  # meters

# Camera (fixed for now)
CAMERA_DISTANCE = 2500  # meters (scene units)
CAMERA_AZIMUTH = 35     # degrees
CAMERA_ELEVATION = 55   # degrees
CAMERA_TARGET = (0, 0, 0)

# =========================
# LOAD DEM
# =========================

def load_dem(path):
    ds = rasterio.open(path)
    elevation = ds.read(1).astype(np.float32)
    transform = ds.transform
    return elevation, transform


# =========================
# CREATE TERRAIN MESH
# =========================

def create_terrain(elev, transform):
    h, w = elev.shape

    verts = []
    faces = []

    for y in range(h):
        for x in range(w):
            z = elev[y, x] * VERTICAL_EXAGGERATION
            world_x = transform * (x, y)
            verts.append((world_x[0], world_x[1], z))

    def idx(x, y):
        return y * w + x

    for y in range(h - 1):
        for x in range(w - 1):
            faces.append([
                idx(x, y),
                idx(x + 1, y),
                idx(x + 1, y + 1),
                idx(x, y + 1)
            ])

    mesh = bpy.data.meshes.new("terrain")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("terrain", mesh)
    bpy.context.collection.objects.link(obj)

    return obj


# =========================
# EXTRACT CONTOURS (SIMPLE METHOD)
# =========================

def generate_contours(elev, transform, interval=10):
    contours = []

    min_z = np.min(elev)
    max_z = np.max(elev)

    levels = np.arange(min_z, max_z, interval)

    h, w = elev.shape

    for level in levels:
        for y in range(h - 1):
            for x in range(w - 1):
                square = elev[y:y+2, x:x+2]

                if np.min(square) <= level <= np.max(square):
                    wx, wy = transform * (x, y)
                    contours.append((wx, wy, level * VERTICAL_EXAGGERATION))

    return contours


def create_contour_object(points):
    mesh = bpy.data.meshes.new("contours")
    obj = bpy.data.objects.new("contours", mesh)
    bpy.context.collection.objects.link(obj)

    verts = points
    edges = [(i, i+1) for i in range(len(points)-1)]

    mesh.from_pydata(verts, edges, [])
    mesh.update()

    return obj


# =========================
# LOAD GPX
# =========================

def load_gpx(path):
    with open(path, 'r') as f:
        gpx = gpxpy.parse(f)

    points = []

    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                points.append((p.longitude, p.latitude, p.elevation or 0))

    return points


# =========================
# PROJECT GPX ON TERRAIN
# =========================

def project_gpx(points, elev_obj):
    terrain = elev_obj.data
    obj = bpy.data.objects["terrain"]

    projected = []

    for x, y, z in points:
        # Ray cast downward
        origin = Vector((x, y, 5000))
        direction = Vector((0, 0, -1))

        result, location, normal, index, hit_obj, matrix = obj.ray_cast(origin, direction)

        if result:
            projected.append(location)

    return projected


def create_gpx_curve(points):
    curve = bpy.data.curves.new('gpx_curve', type='CURVE')
    curve.dimensions = '3D'

    polyline = curve.splines.new('POLY')
    polyline.points.add(len(points)-1)

    for i, p in enumerate(points):
        polyline.points[i].co = (p[0], p[1], p[2], 1)

    obj = bpy.data.objects.new('GPX', curve)
    bpy.context.collection.objects.link(obj)

    return obj


# =========================
# CAMERA SETUP (FIXED)
# =========================

def setup_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.object

    az = radians(CAMERA_AZIMUTH)
    el = radians(CAMERA_ELEVATION)

    cam.location = (
        CAMERA_DISTANCE * np.cos(el) * np.cos(az),
        CAMERA_DISTANCE * np.cos(el) * np.sin(az),
        CAMERA_DISTANCE * np.sin(el)
    )

    cam.data.clip_end = 50000
    cam.data.clip_start = 1

    bpy.context.scene.camera = cam

    return cam


# =========================
# MATERIALS (WHITE LINES ON BLACK)
# =========================

def setup_materials():
    mat = bpy.data.materials.new(name="lines")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in nodes:
        nodes.remove(n)

    output = nodes.new(type='ShaderNodeOutputMaterial')
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[0].default_value = (1, 1, 1, 1)
    emission.inputs[1].default_value = 5.0

    links.new(emission.outputs[0], output.inputs[0])

    return mat


# =========================
# MAIN PIPELINE
# =========================

def main():
    elev, transform = load_dem(DEM_PATH)

    terrain = create_terrain(elev, transform)

    contours = generate_contours(elev, transform, CONTOUR_INTERVAL)
    contour_obj = create_contour_object(contours)

    gpx_points = load_gpx(GPX_PATH)

    cam = setup_camera()
    mat = setup_materials()

    terrain.active_material = mat
    contour_obj.active_material = mat

    print("Scene generated")


main()