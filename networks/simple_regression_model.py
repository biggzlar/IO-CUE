import torch
import torch.nn as nn

class SimpleRegressionModel(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=16, output_dim=1, activation=nn.ReLU, return_activations=False):
        """
        Base neural network that can be used for both mean and variance prediction
        
        Args:
            input_dim: Input dimension 
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (usually 1)
            activation: Activation function to use (default: Mish)
        """
        super(SimpleRegressionModel, self).__init__()
        self.return_activations = return_activations


        self.layer_1 = nn.Linear(in_channels, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation()

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(output_dim)
        ])
    
    def forward(self, x):
        """Forward pass through the network"""
        x_1 = self.layer_1(x)
        z_1 = self.activation(x_1)
        x_2 = self.layer_2(z_1)
        z_2 = self.activation(x_2)
        x_3 = self.layer_3(z_2)
        z_3 = self.activation(x_3)
        outs = [head(z_3) for head in self.heads]
        activations = [x_1, x_2, x_3]
        if self.return_activations:
            return torch.concat(outs, dim=1), activations, None
        else:
            return torch.concat(outs, dim=1)
