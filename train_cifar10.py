import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Internal imports
import nets, diffusionmodel
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=160)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])  # Only for DDPM sampler
    parser.add_argument("--logistic_params", type=tuple, default=(2., 3.))  # lognsr location and scale parameters
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--embedding", type=bool, default=True, help="Use sinusoidal embedding for input and snr/time")
    config = parser.parse_args()

    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)

    # Data
    train = CIFAR10(root='./data', train=True, download=True)
    test = CIFAR10(root='./data', train=False, download=True)

    train_dl = DataLoader(train, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(test, batch_size=config.train_batch_size, shuffle=False, drop_last=True)

    # Model
    denoiser = nets.MLP(in_dim=2,
                        hidden_dim=config.hidden_size,
                        n_layers=config.hidden_layers)
    dm = diffusionmodel.DiffusionModel(denoiser,
                                       x_shape=(2,),
                                       learning_rate=config.learning_rate,
                                       logsnr_loc=config.logistic_params[0], logsnr_scale=config.logistic_params[1])

    trainer = pl.Trainer(max_epochs=config.num_epochs, enable_checkpointing=True, accelerator=device,
                         default_root_dir=outdir, callbacks=[MyCallback()])
    trainer.fit(dm, train_dl, val_dl)