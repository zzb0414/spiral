"""
Spiral deblurring ResNet classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A basic 3-layer ResNet.
input channels: Real, Imag
f[0] * f[0] * n[0]
ReLU
f[1] * f[1] * n[1]
ReLU
1 * 1 * 2
output channels: Real, Imag
"""
class ResidualCNN_3layers(nn.Module):
    def __init__(self, f, n):
        super(ResidualCNN_3layers, self).__init__()
        
        # Layer 1: conv2D (n1=64 filters, f1 x f1 kernel, 2 input channels)
        # Note: Padding is often used to keep the spatial dimensions the same 
        # so the skip connection addition works.
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=n[0], kernel_size=f[0], padding=f[0]//2)
        
        # Layer 2: conv2D (n2=32 filters, f2 x f2 kernel, 64 input channels)
        self.conv2 = nn.Conv2d(in_channels=n[0], out_channels=n[1], kernel_size=f[1], padding=f[1]//2)
        
        # Layer 3: conv2D (n3=2 filters, 1 x 1 kernel, 32 input channels)
        # We don't apply ReLU here because it's the final layer before the sum
        self.conv3 = nn.Conv2d(in_channels=n[1], out_channels=2, kernel_size=1)

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


"""
A basic 5-layer ResNet.
input channels: Real, Imag
ii-th layer, ii from 0 to 3
f[ii] * f[ii] * n[ii]
ReLU
f[-1] * f[-1] * 2
output channels: Real, Imag
"""
class ResidualCNN_5layers(nn.Module):
    def __init__(self, f, n):
        super(ResidualCNN_5layers, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=n[0], kernel_size=f[0], padding=f[0]//2)
        self.conv2 = nn.Conv2d(in_channels=n[0], out_channels=n[1], kernel_size=f[1], padding=f[1]//2)
        self.conv3 = nn.Conv2d(in_channels=n[1], out_channels=n[2], kernel_size=f[2], padding=f[2]//2)
        self.conv4 = nn.Conv2d(in_channels=n[2], out_channels=n[3], kernel_size=f[3], padding=f[3]//2)
        self.conv5 = nn.Conv2d(n[3], 2, kernel_size=f[4], padding=f[4]//2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out) # No ReLU here
        out += identity
        
        return out


"""
A 5-layer ResNet with 3 input channels.
input channels: Real, Imag, B0
ii-th layer, ii from 0 to 3
f[ii] * f[ii] * n[ii]
ReLU
f[-1] * f[-1] * 2
output channels: Real, Imag
"""
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


"""
A 5-layer ResNet with 4 input channels.
input channels: Real, Imag, Real(Phasor), Imag(Phasor)
ii-th layer, ii from 0 to 3
f[ii] * f[ii] * n[ii]
ReLU
f[-1] * f[-1] * 2
output channels: Real, Imag
"""
class ResidualCNN_5layers_withPhi(nn.Module):
    def __init__(self, f, n):
        super(ResidualCNN_5layers_withPhi, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=n[0], kernel_size=f[0], padding=f[0]//2)
        self.conv2 = nn.Conv2d(in_channels=n[0], out_channels=n[1], kernel_size=f[1], padding=f[1]//2)
        self.conv3 = nn.Conv2d(in_channels=n[1], out_channels=n[2], kernel_size=f[2], padding=f[2]//2)
        self.conv4 = nn.Conv2d(in_channels=n[2], out_channels=n[3], kernel_size=f[3], padding=f[3]//2)
        self.conv5 = nn.Conv2d(n[3], 2, kernel_size=f[4], padding=f[4]//2)

        self.residual_layer = nn.Conv2d(4, 2, kernel_size=1)

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


"""
Multi scale ResNet.
"""
class ResidualCNN_MSResNet_withB0(nn.Module):
    def __init__(self, f, n):
        super(ResidualCNN_MSResNet_withB0, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n[0], kernel_size=f[0], padding=f[0]//2)

        self.branch1 = nn.Conv2d(in_channels=n[0], out_channels=n[1], kernel_size=f[1], padding=f[1]//2)
        self.branch2 = nn.Conv2d(in_channels=n[0], out_channels=n[2], kernel_size=f[2], padding=f[2]//2)
        self.branch3 = nn.Conv2d(in_channels=n[0], out_channels=n[3], kernel_size=f[3], padding=f[3]//2)

        self.fuse = nn.Conv2d(in_channels=n[1]+n[2]+n[3], out_channels=n[4], kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels=n[4], out_channels=2, kernel_size=f[4], padding=f[4]//2)

        self.residual_layer = nn.Conv2d(3, 2, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.residual_layer(x)

        feat = F.relu(self.conv1(x))

        out1 = F.relu(self.branch1(feat))
        out2 = F.relu(self.branch2(feat))
        out3 = F.relu(self.branch3(feat))

        combined = torch.cat([out1, out2, out3], dim=1)
        fused = F.relu(self.fuse(combined))

        out = self.conv_out(fused)
        out += identity

        return out


"""
The train model function.
"""
def train_model(model, train_loader, optimizer, scheduler, device, criterion_l1, weighted_l1_loss, criterion_grad, lambda_grad=0.1, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        running_total_loss = 0.0
        running_l1 = 0.0
        running_grad = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # Individual batch losses
            batch_l1_plain = criterion_l1(outputs, targets)
            batch_l1_weighted = weighted_l1_loss(outputs, targets)
            batch_grad = criterion_grad(outputs, targets, loss_type="MSELoss")
            
            # Total composite loss for backprop
            loss = batch_l1_weighted + lambda_grad * batch_grad
            
            loss.backward()
            optimizer.step()
            
            # Accumulate values
            running_total_loss += loss.item()
            running_l1 += batch_l1_plain.item()
            running_grad += batch_grad.item()
        
        # Calculate final epoch averages
        num_batches = len(train_loader)
        avg_epoch_loss = running_total_loss / num_batches
        avg_l1 = running_l1 / num_batches
        avg_grad = running_grad / num_batches
        
        # Step scheduler based on average total loss
        scheduler.step(avg_epoch_loss)
        
        # Print averages for the epoch
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # Crucial for LR consistency
            }
            torch.save(checkpoint, 'models/checkpoint.pth')
            print(f"Epoch [{epoch+1}/{epochs}], Avg L1: {avg_l1:.6f}, Avg Grad: {avg_grad:.6f}, LR: {current_lr:.6e}")

        # Initialize best_l1 if first epoch
        if epoch == 0:
            best_l1 = avg_l1
            
        # Checkpoint based on average plain L1
        if avg_l1 < best_l1:
            best_l1 = avg_l1
            torch.save(model.state_dict(), 'models/best_intermediate_model.pth')
            print(f"--- New Best Average L1 found: {best_l1:.6f} at Epoch {epoch} ---")

    print("Training Complete")