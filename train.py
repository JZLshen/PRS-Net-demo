import torch
import torch.optim as optim
import trimesh
import numpy as np

import config
from model import PRSNet
from loss import SymmetryLoss
from utils import preprocess_mesh, validate_planes, validate_axes


def initialize_weights(model):
    """
    根据论文描述，初始化权重以使预测的平面/轴初始时相互正交。
    这通过初始化每个独立预测头的最后一层权重来实现。
    """
    with torch.no_grad():
        # 为平面和轴创建正交的初始法向量/轴向量
        init_vectors = torch.eye(3, device=next(model.parameters()).device)

        # 分别初始化平面和轴的预测头
        for i, head in enumerate(model.plane_heads):
            last_layer = head.fc_layers[-1]
            # 将前3个输出（法向量）的权重设置为正交
            torch.nn.init.zeros_(last_layer.weight)
            last_layer.weight[:3, :] = (
                torch.randn_like(last_layer.weight[:3, :]) * 0.01
            )  # 随机小值
            last_layer.weight[:3, -16:] = (
                init_vectors[i].repeat(16, 1).t() * 0.1
            )  # 尝试影响输出

            if last_layer.bias is not None:
                torch.nn.init.zeros_(last_layer.bias)
                last_layer.bias[:3] = init_vectors[i]

        for i, head in enumerate(model.axis_heads):
            last_layer = head.fc_layers[-1]
            # (w, x, y, z) -> w=0, (x,y,z)正交
            torch.nn.init.zeros_(last_layer.weight)
            last_layer.weight[1:, :] = torch.randn_like(last_layer.weight[1:, :]) * 0.01
            last_layer.weight[1:, -16:] = init_vectors[i].repeat(16, 1).t() * 0.1

            if last_layer.bias is not None:
                torch.nn.init.zeros_(last_layer.bias)
                last_layer.bias[1:] = init_vectors[i]


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    voxel_input, surface_points, closest_point_grid, grid_min, grid_max = (
        preprocess_mesh(mesh, config.VOXEL_RESOLUTION, config.N_SAMPLE_POINTS)
    )

    voxel_input = voxel_input.to(device)
    surface_points = surface_points.to(device)
    closest_point_grid = closest_point_grid.to(device)
    grid_min = grid_min.to(device)
    grid_max = grid_max.to(device)
    model = PRSNet().to(device)

    initialize_weights(model)

    criterion = SymmetryLoss(w_r=config.W_R)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()

        pred_planes, pred_axes = model(voxel_input)

        loss, loss_sd, loss_r = criterion(
            pred_planes,
            pred_axes,
            surface_points,
            closest_point_grid,
            grid_min,
            grid_max,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {loss.item():.6f}, SD_Loss: {loss_sd.item():.6f}, R_Loss: {loss_r.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        final_pred_planes, final_pred_axes = model(voxel_input)

    print("\nRaw predicted plane parameters:")
    print(final_pred_planes[0].cpu().numpy())

    print("\nRaw predicted axis (quaternion) parameters:")
    print(final_pred_axes[0].cpu().numpy())

    validated_planes = validate_planes(
        final_pred_planes[0],
        surface_points,
        config.VALIDATION_ERROR_THRESHOLD,
        config.VALIDATION_ANGLE_THRESHOLD_RAD,
    )

    print(f"\nValidated planes ({len(validated_planes)} found):")
    if validated_planes:
        for plane in validated_planes:
            print(plane)
    else:
        print("No valid symmetry planes found.")
    validated_axes = validate_axes(final_pred_axes[0], surface_points,
                                    config.VALIDATION_ERROR_THRESHOLD,
                                    config.VALIDATION_ANGLE_THRESHOLD_RAD)

    print(f"\nValidated axes ({len(validated_axes)} found):")
    if validated_axes:
        for axis in validated_axes: print(axis)
    else:
        print("No valid symmetry axes found.")


if __name__ == "__main__":
    main()
