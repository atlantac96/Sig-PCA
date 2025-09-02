import torch
from torch import nn
import torch.nn.functional as F


def custom_loss(prediction, target, nwpstacktensor, lambda_penalty=10):
    mse_loss = F.mse_loss(prediction, target)

    # Calculate NWP + OUTPUT to check for negative values
    nwp_plus_output = nwpstacktensor + prediction
    
    # Penalize cases where NWP + OUTPUT is negative
    if (nwp_plus_output < 0).any():
        penalty = lambda_penalty * torch.mean((nwp_plus_output[nwp_plus_output < 0]) ** 2)
    else:
        penalty = 0.0
    
    # Return the total loss as the base loss plus any penalty
    return mse_loss + penalty

    
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, output_size)
        

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        
        x = torch.relu(self.bn2(self.fc2(x)))
        
        x = self.fc3(x)  # No activation for regression
        return x

class SimpleNNpos(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNNpos, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, output_size)
        

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        
        x = self.fc3(x)  # No activation for regression
        
        return torch.relu(x)  # Ensures non-negative output