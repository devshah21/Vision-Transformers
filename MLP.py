import torch
import torch.nn as nn

class MLP(nn.Module):
    # Define a custom Multi-Layer Perceptron (MLP) neural network module.

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        # Constructor for the MLP class.
        # Parameters:
        # - in_features: Number of input features (input dimensions).
        # - hidden_features: Number of nodes or units in the hidden layer.
        # - out_features: Number of output features (output dimensions).
        # - p: Dropout probability (default is 0, meaning no dropout by default).

        # Call the constructor of the parent class (nn.Module).
        super().__init()

        # Create the first linear layer: in_features -> hidden_features.
        self.fc1 = nn.Linear(in_features, hidden_features)

        # Apply the GELU (Gaussian Error Linear Unit) activation function.
        self.act = nn.GELU()

        # Create the second linear layer: hidden_features -> out_features.
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Apply dropout with the specified probability.
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
    # Forward pass through the MLP network.

        # Apply the first linear transformation (fc1) to the input tensor x.
        x = self.fc1(x)

        # Apply the activation function (GELU) to the result of the first linear layer.
        x = self.act(x)

        # Apply dropout to the output of the activation function.
        x = self.drop(x)

        # Apply the second linear transformation (fc2) to the modified x.
        x = self.fc2(x)

        # Apply dropout to the output of the second linear layer.

        # Return the final output tensor after passing through the network.
        return x