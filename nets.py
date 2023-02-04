import math
import torch as t
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, activation=nn.ReLU, dropout=0., emb_size=128, embedding=True):
        super().__init__()
        self.embedding = embedding
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.time_mlp = SinusoidalEmbedding(emb_size, scale=25.)  # "time" embedding in prev work is 1-1000. Ours is logsnr, ~(-5, 10)
        self.input_mlp1 = SinusoidalEmbedding(emb_size, scale=25.)
        self.input_mlp2 = SinusoidalEmbedding(emb_size, scale=25.)

        if self.embedding:
            self.layers.append(nn.Linear((in_dim + 1) * emb_size, hidden_dim))  # Concatenate, after embeddings
        else:
            self.layers.append(nn.Linear(in_dim + 1, hidden_dim))  # Concatenate logsnr with input, no embedding
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
        if self.embedding:
            x1_emb = self.input_mlp1(x[:, 0])
            x2_emb = self.input_mlp2(x[:, 1])
            t_emb = self.time_mlp(logsnr)
            x = t.cat((x1_emb, x2_emb, t_emb), dim=-1)
        else:
            x = t.concat((x, logsnr.unsqueeze(1)), dim=1)  # concatenate logsnr
        for layer in self.layers:
            x = layer(x)
        return x

# TODO: input embeddings were super useful in https://github.com/tanelp/tiny-diffusion/blob/master/positional_embeddings.py
# Time embeddings were not as important.
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        half_size = self.size // 2
        emb = t.log(t.Tensor([10000.0]).to(x.device)) / (half_size - 1)
        emb = t.exp(-emb * t.arange(half_size, device=x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = t.cat((t.sin(emb), t.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size