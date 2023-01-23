import math
import torch as t
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, activation=nn.ReLU, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim + 1, hidden_dim))  # Concatenate logsnr - should use embedding in high-d
        self.layers.append(activation())
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, in_dim))

    def forward(self, x, logsnr):
        x = t.concat((x, logsnr.unsqueeze(1)), dim=1)  # concatenate logsnr
        for layer in self.layers:
            x = layer(x)
        return x