import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualCNN_5layers_withB0(nn.Module):
    def __init__(self, f, n):
        super(ResidualCNN_5layers_withB0, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n[0], kernel_size=f[0], padding=f[0]//2)
        self.conv2 = nn.Conv2d(in_channels=n[0], out_channels=n[1], kernel_size=f[1], padding=f[1]//2)
        self.conv3 = nn.Conv2d(in_channels=n[1], out_channels=n[2], kernel_size=f[2], padding=f[2]//2)
        self.conv4 = nn.Conv2d(in_channels=n[2], out_channels=n[3], kernel_size=f[3], padding=f[3]//2)
        self.conv5 = nn.Conv2d(n[3], 2, kernel_size=f[4], padding=f[4]//2)

        self.residual_layer = nn.Conv2d(3, 2, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.residual_layer(x)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out) # No ReLU here
        out += identity
        
        return out