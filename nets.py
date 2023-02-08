import numpy as np
import torch as t
import torch.nn as nn
from diffusers import UNet2DModel


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
        self.time_mlp = SinusoidalEmbedding(emb_size, scale=50.)  # "time" embedding in prev work is 1-1000. Ours is logsnr, ~(-5, 10)
        self.input_mlp1 = SinusoidalEmbedding(emb_size, scale=100.)
        self.input_mlp2 = SinusoidalEmbedding(emb_size, scale=100.)

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


class WrapUNet2DModel(UNet2DModel):
    """Wrap UNet2DModel to accept arguments compatible with Diffusion Model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, z, logsnr):
        x = z[0]
        timestep = self.logsnr2t(logsnr)
        eps_hat = super().forward(x, timestep)["sample"]
        return eps_hat

    def logsnr2t(self, logsnr):
        num_diffusion_steps = 10000 # improve the timestep precision
        alphas_cumprod = t.sigmoid(logsnr)
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
        alphas = 1.0 - betas
        alphabarGT = t.tensor(np.cumprod(alphas, axis=0), device=alphas_cumprod.device)
        timestep = t.argmin(abs(alphabarGT-alphas_cumprod[0])) * scale # only use one alphas_cumprod in the batch
        return timestep * t.ones(alphas_cumprod.shape[0], device=alphas_cumprod.device)  # reconvert to batch
