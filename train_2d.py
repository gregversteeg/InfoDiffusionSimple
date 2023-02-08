import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Internal imports
import nets, datasets, diffusionmodel
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons", "scg"])
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=160)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--logsnr_loc", type=float, default=0.)  # lognsr location and scale parameters
    parser.add_argument("--logsnr_scale", type=float, default=3.)  # lognsr location and scale parameters
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--embedding", type=bool, default=True, help="Use sinusoidal embedding for input and snr/time")
    config = parser.parse_args()

    outdir = f"exps/{config.dataset}-{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)

    # Data
    dataset = datasets.get_dataset(config.dataset)
    n = len(dataset)
    train, val = torch.utils.data.random_split(dataset, [int(0.9*n), n - int(0.9*n)])
    train_dl = DataLoader(train, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val, batch_size=(n - int(0.9*n)), shuffle=False, drop_last=False)  # one giant batch

    # Model
    denoiser = nets.MLP(in_dim=2,
                        hidden_dim=config.hidden_size,
                        n_layers=config.hidden_layers)
    dm = diffusionmodel.DiffusionModel(denoiser,
                                       x_shape=(2,),
                                       learning_rate=config.learning_rate,
                                       logsnr_loc=config.logsnr_loc, logsnr_scale=config.logsnr_scale)

    # Train
    x_sample = dataset.tensors[0].numpy()
    random_rows = np.random.choice(len(x_sample), min(200, len(x_sample)), replace=False)
    x_sample = x_sample[random_rows]
    class MyCallback(Callback):
        val_loop_count = 0
        save_grid = []
        grid_x, grid_y = None, None
        def on_train_epoch_end(self, trainer, pl_module):
            # Save contour plot info
            c = 3.
            x_min, x_max, y_min, y_max, r = -c, c, -c, c, 40
            grid_x, grid_y = torch.meshgrid(torch.linspace(x_min, x_max, r), torch.linspace(y_min, y_max, r), indexing='ij')
            xs = torch.stack([grid_x.flatten(), grid_y.flatten()]).T
            xs = xs.to(device)
            with torch.no_grad():
                nll_grid = - torch.stack([pl_module.nll_x(xs[i]) for i in range(len(xs))]).reshape((r, r)).cpu().numpy()

            self.val_loop_count += 1
            self.save_grid.append(nll_grid)
            self.grid_x, self.grid_y = grid_x, grid_y

        def on_train_end(self, trainer, pl_module):
            print("training is ending")
            trainer.save_checkpoint(f"{outdir}/model.pth")
            # new_model = MyLightningModule.load_from_checkpoint(checkpoint_path=f"{outdir}/model.pth")

            # Contour plots
            n = len(self.save_grid)
            k = 8  # max figures
            multiple = max(n // k, 1)
            fig, axs = plt.subplots(1, k, figsize=(10*k, 10))
            # cs = axs[-1].contourf(self.grid_x, self.grid_y, self.save_grid[-1])
            # levels = cs.levels
            levels = [-5., -4., -3., -2., -1.]

            for i, ax in enumerate(axs):
                this_epoch = (i+1) * multiple - 1
                ax.set_title(f'Epoch {this_epoch}')
                ax.set_ylabel('$x_1$')
                ax.set_xlabel('$x_2$')
                cs = ax.contourf(self.grid_x, self.grid_y, self.save_grid[this_epoch], levels, extend='max')
                ax.scatter(x_sample[:, 0], x_sample[:,1], s=40, c='black', alpha=0.6)
            cbar = fig.colorbar(cs, ax=ax)
            cbar.ax.set_label(['$-\log p(x)$'])
            fig.tight_layout()

            # log and save contour plot figure
            tb = pl_module.logger.experiment  # tensorboard logger
            tb.add_figure('contours', figure=fig)
            fig.savefig(f"{imgdir}/contours.png")

    trainer = pl.Trainer(max_epochs=config.num_epochs, enable_checkpointing=True, accelerator=device,
                         default_root_dir=outdir, callbacks=[MyCallback()])
    trainer.fit(dm, train_dl, val_dl)