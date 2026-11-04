from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    lat: np.ndarray
    lon: np.ndarray
    window: tuple[int, int, int, int]

    @classmethod
    def from_coordinates(cls, lat, lon, lat_min=30.0, lat_max=33.5, lon_min=75.5, lon_max=79.5):
        lat_values = np.asarray(lat, dtype=np.float64)
        lon_values = np.asarray(lon, dtype=np.float64)
        if lat_values.ndim != 1 or lon_values.ndim != 1:
            raise ValueError("lat and lon must be one-dimensional")
        if not np.all(np.diff(lat_values) < 0):
            raise ValueError("lat must be descending")
        lat_indices = np.flatnonzero((lat_values >= lat_min) & (lat_values <= lat_max))
        lon_indices = np.flatnonzero((lon_values >= lon_min) & (lon_values <= lon_max))
        if lat_indices.size != 15 or lon_indices.size != 17:
            raise ValueError(f"expected a 15x17 window, got {lat_indices.size}x{lon_indices.size}")
        i0, i1 = int(lat_indices[0]), int(lat_indices[-1] + 1)
        j0, j1 = int(lon_indices[0]), int(lon_indices[-1] + 1)
        return cls(lat_values, lon_values, (i0, i1, j0, j1))

    @property
    def window_shape(self):
        i0, i1, j0, j1 = self.window
        return i1 - i0, j1 - j0
