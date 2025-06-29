import torch
import torch.nn.functional as F
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def create_closest_point_grid(mesh, grid_resolution):
    surface_points, _ = trimesh.sample.sample_surface(mesh, 100000)
    kdtree = cKDTree(surface_points)
    grid_min, grid_max = mesh.bounds
    axis_coords = [np.linspace(grid_min[i], grid_max[i], grid_resolution) for i in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(*axis_coords, indexing='ij')
    grid_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    _, indices = kdtree.query(grid_points)
    closest_points = surface_points[indices]
    closest_point_grid = closest_points.reshape(grid_resolution, grid_resolution, grid_resolution, 3)
    return torch.from_numpy(closest_point_grid.astype(np.float32)), torch.from_numpy(grid_min.astype(np.float32)), torch.from_numpy(grid_max.astype(np.float32))

def preprocess_mesh(mesh, voxel_resolution, n_points):
    scale = 1.0 / np.max(mesh.extents)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.centroid)
    voxel_grid = mesh.voxelized(pitch=1.0 / voxel_resolution).fill()
    voxel_tensor = torch.from_numpy(voxel_grid.matrix.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points_tensor = torch.from_numpy(points.astype(np.float32))
    closest_point_grid, grid_min, grid_max = create_closest_point_grid(mesh, voxel_resolution)
    return voxel_tensor, points_tensor, closest_point_grid, grid_min, grid_max

def validate_planes(pred_planes, surface_points, error_threshold, angle_threshold_rad):
    valid_planes, plane_errors = [], []
    for plane_params in pred_planes:
        normal, d = plane_params[:3], plane_params[3]
        n = F.normalize(normal.unsqueeze(0), p=2, dim=1).squeeze(0)
        proj = torch.sum(surface_points * n, dim=1) + d
        reflected_points = surface_points - 2 * proj.unsqueeze(1) * n.unsqueeze(0)
        dist_matrix = torch.cdist(reflected_points, surface_points)
        min_dists, _ = torch.min(dist_matrix, dim=1)
        error = torch.mean(min_dists).item()
        if error < error_threshold:
            valid_planes.append(plane_params.detach().cpu().numpy())
            plane_errors.append(error)

    if not valid_planes:
        return []

    final_planes, used_indices = [], [False] * len(valid_planes)
    sorted_indices = np.argsort(plane_errors)
    for i in range(len(valid_planes)):
        idx1 = sorted_indices[i]
        if used_indices[idx1]: continue
        final_planes.append(valid_planes[idx1])
        used_indices[idx1] = True
        normal1 = F.normalize(torch.from_numpy(valid_planes[idx1][:3]), p=2, dim=0).numpy()
        for j in range(i + 1, len(valid_planes)):
            idx2 = sorted_indices[j]
            if used_indices[idx2]: continue
            normal2 = F.normalize(torch.from_numpy(valid_planes[idx2][:3]), p=2, dim=0).numpy()
            angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
            if angle < angle_threshold_rad or angle > (np.pi - angle_threshold_rad):
                used_indices[idx2] = True
    return final_planes