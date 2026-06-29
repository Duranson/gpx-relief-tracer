from pathlib import Path

import numpy as np
import pytest

from blender_tracer import select_dem_for_gpx, DEM_FOLDER


def test_select_dem_prefers_overlapping_tile(tmp_path):
    # Minimal synthetic DEM metadata with known bounds in Lambert 93.
    dem_a = tmp_path / 'tile_a.asc'
    dem_b = tmp_path / 'tile_b.asc'

    dem_a.write_text('ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n1 2\n3 4\n', encoding='utf-8')
    dem_b.write_text('ncols 2\nnrows 2\nxllcorner 1000\nyllcorner 1000\ncellsize 10\nNODATA_value -9999\n1 2\n3 4\n', encoding='utf-8')

    gpx_x = np.array([50.0, 80.0], dtype=np.float64)
    gpx_y = np.array([50.0, 80.0], dtype=np.float64)

    selected = select_dem_for_gpx([dem_a, dem_b], gpx_x, gpx_y)

    assert selected == dem_a

# Test the function on a temporary directory with synthetic DEM tiles and GPX coordinates to ensure it selects the correct overlapping tile.
tmp_folder = Path(DEM_FOLDER) / 'test_dem_selection'
tmp_folder.mkdir(parents=True, exist_ok=True)

test_select_dem_prefers_overlapping_tile(tmp_folder)

# Clean up the temporary folder with its files after the test
for tmp_file in tmp_folder.iterdir():
    tmp_file.unlink()
tmp_folder.rmdir()