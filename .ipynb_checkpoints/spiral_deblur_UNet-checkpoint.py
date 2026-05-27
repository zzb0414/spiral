"""
Spiral deblurring U-Net classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A generic and modular multi scale residual block.
"""
class MS_block(nn.Module):
    def __init__(self, in_ch, out_ch, f):
        super(MS_block, self).__init__()

        self.branch1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=f[0], padding=f[0]//2)
        self.branch2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=f[1], padding=f[1]//2)
        self.branch3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=f[2], padding=f[2]//2)

        self.fuse = nn.Conv2d(in_channels=3*out_ch, out_channels=out_ch, kernel_size=1)

        if in_ch != out_ch:
            self.residual_layer = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.residual_layer = nn.Identity()
            

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.residual_layer(x)

        out1 = F.relu(self.branch1(x))
        out2 = F.relu(self.branch2(x))
        out3 = F.relu(self.branch3(x))

        combined = torch.cat([out1, out2, out3], dim=1)
        fused = self.fuse(combined)

        return identity + fused


"""
Residual U-Net.
"""
class MS_UNet(nn.Module):
    def __init__(self, f, n):
        super(MS_UNet, self).__init__()
        
        # Encoder Path
        self.enc1 = MS_block(3, n[0], f)    # Input (R, I, B0) -> n[0]
        self.pool = nn.MaxPool2d(2)
        
        self.enc2 = MS_block(n[0], n[1], f)  # Bottleneck
        
        # Decoder Path
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Input to dec1 is sum of n[0] and n[1] -> Output n[0] due to symmetry
        self.dec1 = MS_block(n[0]+n[1], n[0], f) 
        
        # Final projection to 2 channels (Real/Imag)
        self.final = nn.Conv2d(n[0], 2, kernel_size=1)
        
        # Global Residual projection (3 -> 2)
        self.global_res = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x):
        # 1. Global shortcut
        global_identity = self.global_res(x)
        
        # 2. Encoder
        s1 = self.enc1(x)       # [B, 3, H, W] -> [B, n[0], H, W]
        p1 = self.pool(s1)      # [B, n[0], H, W] -> [B, n[0], H/2, W/2]
        
        b = self.enc2(p1)       # [B, n[0], H/2, W/2] -> [B, n[1], H/2, W/2]
        
        # 3. Decoder
        up1 = self.up(b)        # [B, n[1], H, W]
        cat1 = torch.cat([up1, s1], dim=1) # [B, n[0]+n[1], H, W]
        d1 = self.dec1(cat1)    # [B, n[0], H, W]
        
        # 4. Final Addition
        # The U-Net generates the 'deblurring correction' 
        # which is added to the projected input image.
        out = self.final(d1) + global_identity # [B, 2, H, W]
        return out