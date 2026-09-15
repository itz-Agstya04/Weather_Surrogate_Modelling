"""
Terrain Pipeline for Himachal Pradesh Microclimates PINO Project.

Features:
- Extracts surface elevation from ERA5 / WeatherBench2 geopotential_at_surface
  for the Himachal Pradesh bounding box (lat: 33.5 to 30.0, lon: 75.5 to 79.5).
- Computes 3-channel terrain features:
    Channel 0: Normalized Elevation
    Channel 1: Slope Magnitude (finite differences)
    Channel 2: Terrain Curvature / Aspect (Laplacian)
- Computes `steep_mask` (steep complex terrain cells above median slope)
  and `node_flat_indices`.
- Builds sparse 8-connected grid graph over steep cells (edge_index, edge_attr).
- Builds dense 8-connected full-grid graph over all 15x17 cells (dense_edge_index, dense_edge_attr).
- Saves everything to DATASET/terrain_hp.npz and runs self-validation.
"""

import os
import numpy as np
import xarray as xr
import torch


ERA5_ZARR_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
LAT_SLICE = slice(33.5, 30.0)
LON_SLICE = slice(75.5, 79.5)


def build_grid_graph(node_coords: list[tuple[int, int]], H: int = 15, W: int = 17):
    """
    Builds an 8-connected grid neighborhood graph over the specified node coordinates.
    Returns:
        edge_index: (2, E) int64
        edge_attr: (E, 2) float32 with normalized coordinate offsets (dr / H, dc / W)
    """
    coord_to_idx = {coord: idx for idx, coord in enumerate(node_coords)}
    
    src_list = []
    dst_list = []
    attr_list = []
    
    # 8-connected neighbor offsets
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for idx, (r, c) in enumerate(node_coords):
        for dr, dc in neighbors:
            neighbor_coord = (r + dr, c + dc)
            if neighbor_coord in coord_to_idx:
                neighbor_idx = coord_to_idx[neighbor_coord]
                src_list.append(idx)
                dst_list.append(neighbor_idx)
                attr_list.append([dr / float(H), dc / float(W)])
                
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)
    return edge_index, edge_attr


def generate_terrain_artifact(output_path: str = "DATASET/terrain_hp.npz") -> dict:
    print(f"\n=======================================================", flush=True)
    print(f"       GENERATING HIMACHAL PRADESH TERRAIN ARTIFACT    ", flush=True)
    print(f"=======================================================", flush=True)
    
    # 1. Fetch surface geopotential from WB2 ERA5
    print("Fetching surface geopotential from WeatherBench2...", flush=True)
    cloud_ds = xr.open_zarr(
        ERA5_ZARR_URL,
        consolidated=True,
        storage_options={"token": "anon"}
    )
    
    z_hp = cloud_ds["geopotential_at_surface"].sel(
        latitude=LAT_SLICE,
        longitude=LON_SLICE
    ).values.astype(np.float32)
    
    # Elevation in meters: z / g (g = 9.80665 m/s^2)
    elevation = z_hp / 9.80665
    H, W = elevation.shape
    assert (H, W) == (15, 17), f"Unexpected terrain shape: {(H, W)} vs (15, 17)"
    print(f"HP Grid Dimensions: {H}x{W} (255 cells)", flush=True)
    print(f"Elevation Range: min={elevation.min():.1f}m, max={elevation.max():.1f}m, mean={elevation.mean():.1f}m", flush=True)
    
    # 2. Compute Terrain Derivatives
    # dy: latitude spacing ~0.25 deg = 0.25 * 111,000m ~ 27,750m
    # dx: longitude spacing ~0.25 deg * cos(31.75 deg) ~ 23,600m
    dy = 27750.0
    dx = 23600.0
    
    grad_y, grad_x = np.gradient(elevation, dy, dx)
    slope = np.sqrt(grad_y**2 + grad_x**2)
    
    # Curvature (Laplacian)
    lap_y, _ = np.gradient(grad_y, dy, dx)
    _, lap_x = np.gradient(grad_x, dy, dx)
    curvature = lap_y + lap_x
    
    # Normalize features to ~zero mean, unit variance
    elev_norm = (elevation - elevation.mean()) / (elevation.std() + 1e-8)
    slope_norm = (slope - slope.mean()) / (slope.std() + 1e-8)
    curv_norm = (curvature - curvature.mean()) / (curvature.std() + 1e-8)
    
    terrain_features = np.stack([elev_norm, slope_norm, curv_norm], axis=0).astype(np.float32)  # (3, 15, 17)
    
    # 3. Define steep_mask and sparse node indices
    # Steep cells: cells with slope greater than median slope (top 50% steepest terrain)
    slope_threshold = float(np.median(slope))
    steep_mask = (slope >= slope_threshold)  # (15, 17) bool
    node_flat_indices = np.where(steep_mask.flatten())[0].astype(np.int64)
    N_sparse = len(node_flat_indices)
    print(f"Steep Terrain Mask: {N_sparse}/{H*W} cells ({100.0 * N_sparse / (H*W):.1f}%)", flush=True)
    
    # Coordinates of sparse nodes
    sparse_coords = [(idx // W, idx % W) for idx in node_flat_indices]
    
    # 4. Build Graphs
    # Sparse 8-connected graph
    edge_index, edge_attr = build_grid_graph(sparse_coords, H, W)
    print(f"Sparse Graph: {N_sparse} nodes, {edge_index.shape[1]} directed edges", flush=True)
    
    # Dense 8-connected full-grid graph (all 255 cells)
    full_coords = [(r, c) for r in range(H) for c in range(W)]
    dense_edge_index, dense_edge_attr = build_grid_graph(full_coords, H, W)
    print(f"Dense Graph:  {H*W} nodes, {dense_edge_index.shape[1]} directed edges", flush=True)
    
    # 5. Save Artifact
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        terrain_features=terrain_features,
        steep_mask=steep_mask,
        node_flat_indices=node_flat_indices,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dense_edge_index=dense_edge_index,
        dense_edge_attr=dense_edge_attr,
    )
    print(f"[SUCCESS] Saved terrain artifact to {output_path}!", flush=True)
    
    # 6. Validate Artifact
    data = np.load(output_path)
    assert "terrain_features" in data and data["terrain_features"].shape == (3, 15, 17)
    assert "steep_mask" in data and data["steep_mask"].shape == (15, 17)
    assert "node_flat_indices" in data and len(data["node_flat_indices"]) == N_sparse
    assert "edge_index" in data and data["edge_index"].shape[0] == 2
    assert "edge_attr" in data and data["edge_attr"].shape == (edge_index.shape[1], 2)
    assert "dense_edge_index" in data and data["dense_edge_index"].shape[0] == 2
    assert "dense_edge_attr" in data and data["dense_edge_attr"].shape == (dense_edge_index.shape[1], 2)
    print("[OK] All 7 keys and tensor shapes verified!", flush=True)
    
    return data


if __name__ == "__main__":
    generate_terrain_artifact()
