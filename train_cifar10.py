"""Minimal example of training on CIFAR-10, coming soon."""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import pytorch_lightning as pl
from diffusers import UNet2DModel

# Internal imports
from nets import WrapUNet2DModel
import diffusionmodel
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device  # For M1/M2 Macs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="cifar10")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--logistic_params", type=tuple, default=(10., 5.))  # lognsr location and scale parameters
    config = parser.parse_args()

    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)

    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))])

    train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_dl = DataLoader(train, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=12)
    val_dl = DataLoader(test, batch_size=config.train_batch_size, shuffle=False, drop_last=True, num_workers=12)


    # Model
    model_id = "google/ddpm-cifar10-32"
    model_pt = UNet2DModel.from_pretrained(model_id)  # Get architecture from pretrained model
    denoiser = WrapUNet2DModel(**model_pt.config)  # Load configuration from pretrained model
    if config.pretrained:  # Load pretrained weights
        denoiser.load_state_dict(model_pt.state_dict())
    denoiser.to(device)

    dm = diffusionmodel.DiffusionModel(denoiser,
                                       x_shape=(3, 32, 32),
                                       learning_rate=config.learning_rate,
                                       logsnr_loc=config.logistic_params[0], logsnr_scale=config.logistic_params[1])

    trainer = pl.Trainer(max_epochs=config.num_epochs, enable_checkpointing=True,
                         accelerator=device, default_root_dir=outdir)
    trainer.fit(dm, train_dl, val_dl)
