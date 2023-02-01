import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

# Internal imports
import nets, datasets, diffusionmodel
device =


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])  # Only for DDPM sampler
    parser.add_argument("--logistic_params", type=tuple, default=(2., 3.))  # lognsr location and scale parameters
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    # parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    # parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

    # Data
    dataset = datasets.get_dataset(config.dataset)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    # Model
    denoiser = nets.MLP(in_dim=2,
                        hidden_dim=config.hidden_size,
                        n_layers=config.hidden_layers)
    dm = diffusionmodel.DiffusionModel(denoiser,
                                       x_shape=(2,),
                                       learning_rate=config.learning_rate
                                       logsnr_loc=2., logsnr_scale=3.)  # TODO

    # Train
    trainer = pl.Trainer(max_epochs=config.num_epochs, enable_checkpointing=True, accelerator=device)
    trainer.fit(dm, train_dl, val_dl)