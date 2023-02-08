import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

# Internal imports
from nets import WrapUNet2DModel
import diffusionmodel
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="cifar10")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=160)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--logistic_params", type=tuple, default=(6., 5.))  # lognsr location and scale parameters
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
    if config.pretrained:
        model_id = "google/ddpm-cifar10-32"
        denoiser = WrapUNet2DModel.from_pretrained(model_id)
    else:
        # TODO: Do I need to get the hyper-params from the pretrained model?
        denoiser = WrapUNet2DModel()

    dm = diffusionmodel.DiffusionModel(denoiser,
                                       x_shape=(3, 32, 32),
                                       learning_rate=config.learning_rate,
                                       logsnr_loc=config.logistic_params[0], logsnr_scale=config.logistic_params[1])

    trainer = pl.Trainer(max_epochs=config.num_epochs, enable_checkpointing=True,
                         accelerator=device, default_root_dir=outdir)
    trainer.fit(dm, train_dl, val_dl)
