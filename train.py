import torch
import torch.optim as optim
import trimesh
import numpy as np

import config
from model import PRSNet
from loss import SymmetryLoss
from utils import preprocess_mesh, validate_planes

def initialize_weights(model):
    """
    手动正交初始化模型的最后一个全连接层，以打破对称性。
    """
    last_layer = model.fc[-1]
    with torch.no_grad():
        init_normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        torch.nn.init.orthogonal_(last_layer.weight)
        
        if last_layer.bias is not None:
            torch.nn.init.zeros_(last_layer.bias)

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    voxel_input, surface_points, closest_point_grid, grid_min, grid_max = preprocess_mesh(
        mesh, config.VOXEL_RESOLUTION, config.N_SAMPLE_POINTS
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

    print("Starting training with custom weight initialization...")
    for epoch in range(config.EPOCHS):
        model.train()
        pred_params = model(voxel_input)
        loss, loss_sd, loss_r = criterion(
            pred_params, surface_points, closest_point_grid, grid_min, grid_max
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{config.EPOCHS}], Loss: {loss.item():.6f}')

    print("\nTraining finished.")

    model.eval()
    with torch.no_grad():
        final_pred_params = model(voxel_input)[0]
    
    print("\nRaw predicted plane parameters:")
    for p in final_pred_params: print(p.cpu().numpy())

    validated_planes = validate_planes(final_pred_params, surface_points,
                                       config.VALIDATION_ERROR_THRESHOLD,
                                       config.VALIDATION_ANGLE_THRESHOLD_RAD)

    print(f"\nValidated planes ({len(validated_planes)} found):")
    if validated_planes:
        for plane in validated_planes: print(plane)
    else:
        print("No valid symmetry planes found.")

if __name__ == '__main__':
    main()