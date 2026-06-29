# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Renders a GPX route as a 3D curve on a stylized elevation map with contour lines inside Blender, producing a static PNG (`render.png`). The terrain is black, contour lines are white emission, and the GPX track is red emission.

## Running the script

The script must run inside Blender's Python environment:

```powershell
# Headless render (adjust path to your Blender installation)
blender --background --python blender_tracer.py

# Interactive: open blender_tracer.py in Blender's Text Editor and press Run Script
```

Outside Blender (for testing pure-Python functions only — Blender scene operations are skipped):

```powershell
.venv\Scripts\python blender_tracer.py
```

Running tests (pytest is installed, no test suite exists yet):

```powershell
.venv\Scripts\pytest
```

## Architecture

Everything lives in the single file [blender_tracer.py](blender_tracer.py). The top of that file holds all configuration constants (`BASE_DIR`, `DEM_FOLDER`, `GPX_PATH`, `VERTICAL_EXAGGERATION`, `CONTOUR_INTERVAL`, camera settings).

### Pipeline (function `main`)

| Step | Functions | Notes |
|------|-----------|-------|
| 1. Clean scene | `clean_scene` | Removes terrain/contours/GPX/camera objects by name prefix |
| 2. Load GPX | `load_gpx` | Returns lon/lat/elev arrays via `gpxpy` |
| 3. Select DEM tiles | `gpx_to_dem_coords`, `discover_dem_candidates`, `select_dem_tiles_for_gpx` | Converts GPX coords from WGS-84 to DEM CRS (EPSG:2154), then selects `.asc` tiles whose bounds overlap any GPX segment |
| 4. Merge & crop DEM | `merge_dem_tiles`, `crop_dem_to_bounds` | Stitches selected tiles into a mosaic, then crops to GPX bounding box + 1 km margin |
| 5. Terrain mesh | `create_terrain` | One Blender vertex per DEM cell; Z = elevation × `VERTICAL_EXAGGERATION` |
| 6. Contour lines | `generate_contour_segments`, `create_contour_object` | Marching-squares per cell per elevation level; result cached as `.npz` in `BASE_DIR` |
| 7. GPX projection | `project_gpx`, `create_gpx_curve` | Ray-cast downward from above terrain; falls back to DEM lookup, then recorded GPX elevation |

After step 7: materials applied, camera positioned, Cycles CPU render written to `render.png`.

### Blender / non-Blender compatibility

`bpy` and `mathutils` are wrapped in `try/except` at import time. A minimal `Vector` stub is defined so functions that only do geometry math (everything except Blender object creation) can be unit-tested with standard Python. Functions that call `bpy` check `if bpy is None: return` at the top.

### DEM data

- Source: French IGN RGE Alti tiles, Lambert 93 projection (EPSG:2154)
- Format: Arc/Info ASCII Grid (`.asc`)
- Location: `contour_lines/isere/ign/` — hundreds of tiles covering the Isère department
- CRS detection: `infer_dem_crs` falls back to filename inspection (looks for `LAMB93`/`IGN69`) if rasterio can't read the CRS from the file header

### Renderable geometry rule

Both Cycles and EEVEE render **surfaces** only. A Blender Mesh with edges but no faces has zero surface area and is invisible in renders (though it appears in the viewport). Any object that must appear in a render — terrain, contour lines, GPX track — needs either mesh faces or a Curve with `bevel_depth` set (which Blender converts to a tube cross-section at render time). The terrain uses a face mesh; the GPX line and contour lines use Curves with `bevel_depth`.

### Contour cache

`contour_cache_path` produces a SHA-256 digest of the selected DEM paths, GPX path, interval, and bounding box. Cache files are written as `contours_cache_<digest>.npz` in `BASE_DIR`. Delete these files to force regeneration.

## Dependencies

Installed in `.venv`: `numpy`, `rasterio`, `gpxpy`, `pytest`.  
Blender-provided (not in `.venv`): `bpy`, `mathutils`.

The files in `old_tests/` are abandoned prototypes (Open Elevation API + matplotlib/pyvista approach) and are not part of the current pipeline.
