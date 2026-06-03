"""
Spiral deblurring U-Net classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A generic and modular multi scale residual block.
f[0] * f[0] * out_ch | f[1] * f[1] * out_ch | f[2] * f[2] * out_ch
1 * 1 conv to fuse the branches
Residual connection
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
1 stage residual U-Net.
        |=======================================================================================|
        |                                                                                       +
3-to-2 conv2D | 3-to-n[0] MS_block ================= (n[1]+n[0])-to-n[0] MS_block --- n[0]-to-2 conv2D
                                | MaxPool by 2      |
                                |                   | Upsample by 2
                                n[0]-to-n[1] MS_block
"""
class MS_UNet(nn.Module):
    def __init__(self, f, n, skip='cat'):
        super(MS_UNet, self).__init__()
        self.skip = skip
        
        # Encoder Path
        self.enc1 = MS_block(3, n[0], f)    # Input (R, I, B0) -> n[0]
        self.pool = nn.MaxPool2d(2)
        
        self.enc2 = MS_block(n[0], n[1], f)  # Bottleneck
        
        # Decoder Path
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Input to dec1 is sum of n[0] and n[1] -> Output n[0] due to symmetry
        if self.skip == 'cat':
            self.dec1 = MS_block(n[0]+n[1], n[0], f)
        elif self.skip == 'add' and n[0] == n[1]:
            self.dec1 = MS_block(n[0], n[0], f)
        else:
            if self.skip == 'add':
                print("Warning: For 'add' skip connection, n[0] and n[1] must be equal. Defaulting to 'cat'.")
                self.skip = 'cat'
            else:
                raise ValueError("Invalid skip connection type. Choose 'cat' or 'add'.")
        
        # Final projection to 2 channels (Real/Imag)
        self.final = nn.Conv2d(n[0], 2, kernel_size=1)
        
        # Global Residual projection (3 -> 2)
        self.global_res = nn.Conv2d(3, 2, kernel_size=1)

        # Initialize Conv2d weights for all submodules in this network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Global shortcut
        # global_identity = self.global_res(x)
        global_identity = x[:, :2, :, :] # Assuming the input channels are ordered as (R, I, B0), we take only R and I for the global residual.
        
        # 2. Encoder
        s1 = self.enc1(x)       # [B, 3, H, W] -> [B, n[0], H, W]
        p1 = self.pool(s1)      # [B, n[0], H, W] -> [B, n[0], H/2, W/2]
        
        b = self.enc2(p1)       # [B, n[0], H/2, W/2] -> [B, n[1], H/2, W/2]
        
        # 3. Decoder
        up1 = self.up(b)        # [B, n[1], H, W]
        if self.skip == 'cat':
            cat1 = torch.cat([up1, s1], dim=1) # [B, n[0]+n[1], H, W]
        else: # 'add'
            cat1 = up1 + s1 # [B, n[0], H, W]
        d1 = self.dec1(cat1)    # [B, n[0], H, W]
        
        # 4. Final Addition
        # The U-Net generates the 'deblurring correction' 
        # which is added to the projected input image.
        out = self.final(d1) + global_identity # [B, 2, H, W]
        return out
    

"""
4 stage residual U-Net.
"""
class MS_UNet_4stage(nn.Module):
    def __init__(self, f, n):
        super(MS_UNet_4stage, self).__init__()
        
        # Encoder Path
        self.pool = nn.MaxPool2d(2)
        self.enc1 = MS_block(3, n[0], f)    # Input (R, I, B0) -> n[0], stage 0
        self.enc2 = MS_block(n[0], n[1], f) # n[0] -> n[1], stage 1
        self.enc3 = nn.Conv2d(in_channels=n[1], out_channels=n[2], kernel_size=3, padding=1) # n[1] -> n[2], stage 2
        self.enc3_res = nn.Conv2d(n[1], n[2], kernel_size=1) # Residual for stage 2
        self.enc4 = nn.Conv2d(in_channels=n[2], out_channels=n[3], kernel_size=3, padding=1) # n[2] -> n[3], stage 3
        self.enc4_res = nn.Conv2d(n[2], n[3], kernel_size=1) # Residual for stage 3
        self.enc5 = nn.Conv2d(in_channels=n[3], out_channels=n[4], kernel_size=3, padding=1) # n[3] -> n[4], stage 4 (bottleneck)
        self.enc5_res = nn.Conv2d(n[3], n[4], kernel_size=1) # Residual for stage 4
        
        # Decoder Path
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Input to decoders is sum of corresponding encoder output and upsampled decoder output from previous stage.
        self.dec4 = nn.Conv2d(in_channels=n[4]+n[3], out_channels=n[3], kernel_size=3, padding=1) # n[4]+n[3] -> n[3], stage 4
        self.dec4_res = nn.Conv2d(n[4]+n[3], n[3], kernel_size=1) # Residual for stage 4
        self.dec3 = nn.Conv2d(in_channels=n[3]+n[2], out_channels=n[2], kernel_size=3, padding=1) # n[3]+n[2] -> n[2], stage 3
        self.dec3_res = nn.Conv2d(n[3]+n[2], n[2], kernel_size=1) # Residual for stage 3
        self.dec2 = MS_block(n[2]+n[1], n[1], f) # n[2]+n[1] -> n[1], stage 2
        self.dec1 = MS_block(n[0]+n[1], n[0], f) 
        
        # Final projection to 2 channels (Real/Imag)
        self.final = nn.Conv2d(n[0], 2, kernel_size=1)
        
        # Global Residual projection (3 -> 2)
        self.global_res = nn.Conv2d(3, 2, kernel_size=1)

        for m in [self.enc3, self.enc3_res, self.enc4, self.enc4_res, self.enc5, self.enc5_res, 
          self.dec4, self.dec4_res, self.dec3, self.dec3_res, self.final, self.global_res]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # 1. Global shortcut
        # global_identity = self.global_res(x)
        global_identity = x[:, :2, :, :] # Assuming the input channels are ordered as (R, I, B0), we take only R and I for the global residual.
        
        # 2. Encoder
        s1 = self.enc1(x)       # [B, 3, H, W] -> [B, n[0], H, W]
        p1 = self.pool(s1)      # [B, n[0], H, W] -> [B, n[0], H/2, W/2]
        s2 = self.enc2(p1)      # [B, n[0], H/2, W/2] -> [B, n[1], H/2, W/2]
        p2 = self.pool(s2)      # [B, n[1], H/2, W/2] -> [B, n[1], H/4, W/4]
        s3_conv = F.leaky_relu(self.enc3(p2))      # [B, n[1], H/4, W/4] -> [B, n[2], H/4, W/4]
        s3 = s3_conv + self.enc3_res(p2) # Residual connection for stage 2
        p3 = self.pool(s3)      # [B, n[2], H/4, W/4] -> [B, n[2], H/8, W/8]
        s4_conv = F.leaky_relu(self.enc4(p3))      # [B, n[2], H/8, W/8] -> [B, n[3], H/8, W/8]
        s4 = s4_conv + self.enc4_res(p3) # Residual connection for stage 3
        p4 = self.pool(s4)      # [B, n[3], H/8, W/8] -> [B, n[3], H/16, W/16] 
        
        b_conv = F.leaky_relu(self.enc5(p4))       # [B, n[3], H/16, W/16] -> [B, n[4], H/16, W/16]
        b = b_conv + self.enc5_res(p4) # Residual connection for stage 4
        
        # 3. Decoder
        up4 = self.up(b)        # [B, n[4], H/8, W/8]
        cat4 = torch.cat([up4, s4], dim=1) # [B, n[3]+n[4], H/8, W/8]
        d4_conv = F.leaky_relu(self.dec4(cat4))    # [B, n[3], H/8, W/8]
        d4 = d4_conv + self.dec4_res(cat4) # Residual connection for stage 4
        up3 = self.up(d4)       # [B, n[3], H/4, W/4]
        cat3 = torch.cat([up3, s3], dim=1) # [B, n[2]+n[3], H/4, W/4]
        d3_conv = F.leaky_relu(self.dec3(cat3))    # [B, n[2], H/4, W/4]
        d3 = d3_conv + self.dec3_res(cat3) # Residual connection for stage 3
        up2 = self.up(d3)       # [B, n[2], H/2, W/2]
        cat2 = torch.cat([up2, s2], dim=1) # [B, n[1]+n[2], H/2, W/2]
        d2 = self.dec2(cat2)    # [B, n[1], H/2, W/2]
        up1 = self.up(d2)       # [B, n[1], H, W]
        cat1 = torch.cat([up1, s1], dim=1) # [B, n[0]+n[1], H, W]
        d1 = self.dec1(cat1)    # [B, n[0], H, W]
        
        # 4. Final Addition
        # The U-Net generates the 'deblurring correction' 
        # which is added to the projected input image.
        out = self.final(d1) + global_identity # [B, 2, H, W]
        return out