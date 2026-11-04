import numpy as np

from data.grid import GridSpec


def test_grid_spec_locates_hp_window():
    lat = np.linspace(45.0, 20.0, 101)
    lon = np.linspace(65.0, 90.0, 101)
    grid = GridSpec.from_coordinates(lat, lon)
    assert grid.window_shape == (15, 17)
