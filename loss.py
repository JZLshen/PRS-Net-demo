import torch
import torch.nn as nn
import torch.nn.functional as F


def quat_rotate(points, q):
    points_quat = F.pad(points, (1, 0), "constant", 0)  # (N, 4)

    q_conj = torch.tensor([q[0], -q[1], -q[2], -q[3]], device=q.device).unsqueeze(0)
    q = q.unsqueeze(0)

    # p' = q * p * q_conj
    rotated_quat = quat_multiply(quat_multiply(q, points_quat), q_conj)
    return rotated_quat[:, 1:]


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)


class SymmetryLoss(nn.Module):
    def __init__(self, w_r):
        super(SymmetryLoss, self).__init__()
        self.w_r = w_r

    def forward(
        self,
        pred_planes,
        pred_axes,
        surface_points,
        closest_point_grid,
        grid_min,
        grid_max,
    ):
        loss_sd_plane = self._calculate_reflection_loss(
            pred_planes[0], surface_points, closest_point_grid, grid_min, grid_max
        )
        loss_sd_axis = self._calculate_rotation_loss(
            pred_axes[0], surface_points, closest_point_grid, grid_min, grid_max
        )
        total_loss_sd = loss_sd_plane + loss_sd_axis

        total_loss_r = self._calculate_regularization_loss(pred_planes[0], pred_axes[0])

        final_loss = total_loss_sd + self.w_r * total_loss_r

        return final_loss, total_loss_sd, total_loss_r

    def _get_closest_point_distance(
        self, transformed_points, closest_point_grid, grid_min, grid_max
    ):
        grid_res = closest_point_grid.shape[0]
        # 将变换后的点归一化到[0, 1]范围，以找到它们在网格中的索引
        normalized_points = (transformed_points - grid_min) / (grid_max - grid_min)
        indices = torch.clamp(
            torch.round(normalized_points * (grid_res - 1)), 0, grid_res - 1
        ).long()

        # 从预计算的网格中获取最近的表面点
        closest_surface_points = closest_point_grid[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ]

        # 计算欧氏距离
        distances = torch.linalg.norm(
            transformed_points - closest_surface_points, dim=1
        )
        return torch.mean(distances)

    def _calculate_reflection_loss(
        self, planes, points, closest_point_grid, grid_min, grid_max
    ):
        loss = 0
        for i in range(planes.size(0)):
            n, d = planes[i, :3], planes[i, 3]
            proj = torch.sum(points * n, dim=1) + d
            reflected_points = points - 2 * proj.unsqueeze(1) * n.unsqueeze(0)
            loss += self._get_closest_point_distance(
                reflected_points, closest_point_grid, grid_min, grid_max
            )
        return loss

    def _calculate_rotation_loss(
        self, axes, points, closest_point_grid, grid_min, grid_max
    ):
        loss = 0
        for i in range(axes.size(0)):
            q = axes[i]
            rotated_points = quat_rotate(points, q)
            loss += self._get_closest_point_distance(
                rotated_points, closest_point_grid, grid_min, grid_max
            )
        return loss

    def _calculate_regularization_loss(self, planes, axes):
        # L_r = ||A||^2_F + ||B||^2_F

        plane_normals = planes[:, :3]
        identity = torch.eye(3, device=planes.device)
        A = torch.matmul(plane_normals, plane_normals.t()) - identity
        loss_r_plane = torch.sum(A**2)

        axis_vectors = axes[:, 1:]  # (w, x, y, z) -> (x, y, z)
        axis_vectors = F.normalize(axis_vectors, p=2, dim=1)
        B = torch.matmul(axis_vectors, axis_vectors.t()) - identity
        loss_r_axis = torch.sum(B**2)

        return loss_r_plane + loss_r_axis
