import torch.nn as nn

class PRSNet(nn.Module):
    """PRS-Net 网络架构 (参考论文 Sec. 3)"""
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
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).view(-1, 3, 4)