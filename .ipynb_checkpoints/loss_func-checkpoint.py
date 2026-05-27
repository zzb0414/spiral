"""
Loss functions classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Weighed L1 norm loss. Heavier penalties on edge regions.
"""
def weighted_l1_loss(outputs, targets):
    # Calculate a simple edge mask from the target
    # Pixels with high gradients get a high weight
    dx = torch.abs(targets[:, :, :, :-1] - targets[:, :, :, 1:])
    dy = torch.abs(targets[:, :, :-1, :] - targets[:, :, 1:, :])
    
    # Pad to match original size
    edge_mask = torch.zeros_like(targets)
    edge_mask[:, :, :, :-1] += dx
    edge_mask[:, :, :-1, :] += dy
    
    # Normalize mask to [1, 10] range
    edge_mask = torch.pow((edge_mask / edge_mask.max()), 0.5) * 9.0 + 1.0
    
    return torch.mean(torch.abs(outputs - targets) * edge_mask)


"""
Gradient loss. Penalizing on differences between targets' and outputs' x- and y-gradient values.
"""
class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, x, y, loss_type='l1_loss'):
        # Calculate horizontal gradients (difference between adjacent columns)
        grad_x_out = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_x_tgt = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        
        # Calculate vertical gradients (difference between adjacent rows)
        grad_y_out = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        grad_y_tgt = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

        if loss_type == 'l1_loss':
            # Return the L1 difference between the gradients
            return F.l1_loss(grad_x_out, grad_x_tgt) + F.l1_loss(grad_y_out, grad_y_tgt)
        elif loss_type == 'MSELoss':
            criterion = nn.MSELoss()
            return criterion(grad_x_out, grad_x_tgt) + criterion(grad_y_out, grad_y_tgt)