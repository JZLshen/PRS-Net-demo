import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionHead(nn.Module):
    def __init__(self):
        super(PredictionHead, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.fc_layers(x)


class PRSNet(nn.Module):
    def __init__(self):
        super(PRSNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(2),
        )

        self.plane_heads = nn.ModuleList([PredictionHead() for _ in range(3)])
        self.axis_heads = nn.ModuleList([PredictionHead() for _ in range(3)])

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # 展平为 (batch, 64)

        plane_outputs = [head(features) for head in self.plane_heads]
        axis_outputs = [head(features) for head in self.axis_heads]

        plane_params = torch.stack(plane_outputs, dim=1)
        axis_params = torch.stack(axis_outputs, dim=1)

        plane_normals = plane_params[:, :, :3]
        plane_distances = plane_params[:, :, 3:]
        normalized_normals = F.normalize(plane_normals, p=2, dim=2)
        planes = torch.cat([normalized_normals, plane_distances], dim=2)

        axes = F.normalize(axis_params, p=2, dim=2)

        return planes, axes
