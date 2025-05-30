import torch
import torch.nn as nn

class SimpleRegressionModel(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=16, output_dim=1, activation=nn.ReLU):
        """
        Base neural network that can be used for both mean and variance prediction
        
        Args:
            input_dim: Input dimension 
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (usually 1)
            activation: Activation function to use (default: Mish)
        """
        super(SimpleRegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(output_dim)
        ])
    
    def forward(self, x):
        """Forward pass through the network"""
        outs = [head(self.network(x)) for head in self.heads]
        return torch.concat(outs, dim=1)
