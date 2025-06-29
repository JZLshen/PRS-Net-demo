import torch
import torch.nn as nn
import torch.nn.functional as F

class PRSNet(nn.Module):
    def __init__(self):
        super(PRSNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.MaxPool3d(2),
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.MaxPool3d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 3 * 4) # 3 planes, 4 params each
        )

    def forward(self, x):
        params = self.encoder(x)
        params = params.view(params.size(0), -1)
        params = self.fc(params).view(-1, 3, 4)
        
        normals = params[:, :, :3]
        distances = params[:, :, 3:]
        
        normalized_normals = F.normalize(normals, p=2, dim=2)
        
        return torch.cat([normalized_normals, distances], dim=2)