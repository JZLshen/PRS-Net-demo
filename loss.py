import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetryLoss(nn.Module):
    def __init__(self, w_r):
        super(SymmetryLoss, self).__init__()
        self.w_r = w_r

    def forward(self, pred_params, surface_points, closest_point_grid, grid_min, grid_max):
        batch_size = pred_params.size(0)
        total_loss_sd = 0
        total_loss_r = 0

        for i in range(batch_size):
            planes = pred_params[i]
            loss_sd = self._calculate_symmetry_distance_loss(planes, surface_points, closest_point_grid, grid_min, grid_max)
            loss_r = self._calculate_regularization_loss(planes)
            total_loss_sd += loss_sd
            total_loss_r += loss_r
        
        avg_loss_sd = total_loss_sd / batch_size
        avg_loss_r = total_loss_r / batch_size

        return avg_loss_sd + self.w_r * avg_loss_r, avg_loss_sd, avg_loss_r
        
    def _calculate_symmetry_distance_loss(self, planes, points, closest_point_grid, grid_min, grid_max):
        loss = 0
        grid_res = closest_point_grid.shape[0]
        for i in range(planes.size(0)):
            n, d = planes[i, :3], planes[i, 3] 
            
            proj = torch.sum(points * n, dim=1) + d
            reflected_points = points - 2 * proj.unsqueeze(1) * n.unsqueeze(0)
            
            normalized_points = (reflected_points - grid_min) / (grid_max - grid_min)
            indices = torch.clamp(torch.round(normalized_points * (grid_res - 1)), 0, grid_res - 1).long()
            
            closest_surface_points = closest_point_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
            distances = torch.linalg.norm(reflected_points - closest_surface_points, dim=1)
            loss += torch.mean(distances)
        return loss

    def _calculate_regularization_loss(self, planes):
        # Regularization Loss (Sec. 4.2)
        normals = F.normalize(planes[:, :3], p=2, dim=1)
        identity = torch.eye(3, device=planes.device)
        A = torch.matmul(normals, normals.t()) - identity
        return torch.sum(A**2)