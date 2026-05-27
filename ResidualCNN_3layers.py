import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualCNN_3layers(nn.Module):
    def __init__(self, f1=9, f2=5, n1=64, n2=32):
        super(ResidualCNN_3layers, self).__init__()
        
        # Layer 1: conv2D (n1=64 filters, f1 x f1 kernel, 2 input channels)
        # Note: Padding is often used to keep the spatial dimensions the same 
        # so the skip connection addition works.
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=n1, kernel_size=f1, padding=f1//2)
        
        # Layer 2: conv2D (n2=32 filters, f2 x f2 kernel, 64 input channels)
        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=f2, padding=f2//2)
        
        # Layer 3: conv2D (n3=2 filters, 1 x 1 kernel, 32 input channels)
        # We don't apply ReLU here because it's the final layer before the sum
        self.conv3 = nn.Conv2d(in_channels=n2, out_channels=2, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store identity for the skip connection
        identity = x
        
        # Block 1
        out = F.relu(self.conv1(x))
        
        # Block 2
        out = F.relu(self.conv2(out))
        
        # Block 3 (Linear output before addition)
        out = self.conv3(out)
        
        # Skip connection: element-wise addition
        out += identity
        
        return out