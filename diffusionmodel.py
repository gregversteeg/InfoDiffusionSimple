"""
Main information theoretic diffusion model class, with optional contrastive loss. See readme.md.
"""
import math
import numpy as np
import torch as t
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class DiffusionModel(pl.LightningModule):
    def __init__(self, denoiser, x_shape=(2,),
                 learning_rate=0.001,
                 logsnr_loc=2., logsnr_scale=3.):  # Log SNR importance sampling distribution parameters
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"])  # saves full argument dict into self.hparams
        self.model = denoiser  # First argument of "model" is data, second is log SNR (per sample)
        self.d = np.prod(x_shape)  # Total dimensionality
        self.h_g = 0.5 * self.d * math.log(2 * math.pi * math.e)  # Differential entropy for N(0,I)
        self.left = (-1,) + (1,) * (len(x_shape))  # View for left multiplying a batch of samples

    def forward(self, x, logsnr):
        return self.model(x, logsnr)

    def training_step(self, batch, batch_idx):
        loss = self.nll(batch)
        self.log("train_loss", loss)
        return loss

    def noisy_channel(self, x, logsnr):
        """Add Gaussian noise to x, return "z" and epsilon."""
        logsnr = logsnr.view(self.left)  # View for left multiplying
        eps = t.randn((len(logsnr),) + self.hparams.x_shape, device=self.device)
        return t.sqrt(t.sigmoid(logsnr)) * x + t.sqrt(t.sigmoid(-logsnr)) * eps, eps

    def mse(self, x, logsnr):
        """MSE for recovering epsilon from noisy channel, for given log SNR values."""
        z, eps = self.noisy_channel(x, logsnr)
        eps_hat = self(z, logsnr)
        error = (eps - eps_hat).flatten(start_dim=1)
        return t.einsum('ij,ij->i', error, error)  # MSE of epsilon estimate, per sample

    def nll(self, batch):
        """Estimate of negative log likelihood for a batch, - E_x [log p(x)], the data distribution."""
        x = batch[0]
        logsnr, weights = logistic_integrate(len(x), self.hparams.logsnr_loc, self.hparams.logsnr_scale,
                                             device=self.device)  # use same device as LightningModule
        mses = self.mse(x, logsnr)
        mmse_gap = mses - self.d * t.sigmoid(logsnr)  # MSE gap compared to using optimal denoiser for N(0,I)
        return self.h_g + 0.5 * (weights * mmse_gap).mean()  # Interpretable as differential entropy (nats)

    def nll_x(self, x, npoints=200):
        """-log p(x) for a single sample, x"""
        return self.nll([x.unsqueeze(0).expand((npoints,) + self.hparams.x_shape)])

    def score(self, x, alpha):
        """\nabla_z \log p_\alpha(z), converges to data dist. score in large SNR limit."""
        return -self.model(x, alpha) / t.sqrt(t.sigmoid(-alpha.view(self.left)))

    def configure_optimizers(self):
        """Pytorch Lightning optimizer hook."""
        return t.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def validation_step(self, batch, batch_idx, nrepeat=20):
        with t.no_grad():
            loss = 0.
            for i in range(nrepeat):
                loss += self.nll(batch) / nrepeat
        self.log("val_loss", loss)
        if batch_idx == 0:  # Plot and log MSE curve, for one batch per epoch
            mses = []
            loc, s = self.hparams.logsnr_loc, self.hparams.logsnr_scale
            x = batch[0]
            logsnrs = t.linspace(loc - 3 * s, loc + 3 * s, 100, device=self.device)
            mmse_g = self.d * t.sigmoid(logsnrs)
            for logsnr in logsnrs:
                mses.append(self.mse(x, t.ones(len(x), device=self.device) * logsnr).mean().cpu())
            tb = self.logger.experiment  # tensorboard logger
            fig, ax = plt.subplots(1, 1)
            ax.plot(logsnrs.cpu(), mses, label="MSE")
            ax.plot(logsnrs.cpu(), mmse_g.cpu(), label='MMSE Gaussian')
            ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
            ax.set_xlabel('log SNR ($\\alpha$)')
            ax.legend()

            tb.add_figure('mses', figure=fig, global_step=self.current_epoch)
        return loss


def logistic_integrate(npoints, loc, scale, clip=3., device='cpu'):
    """Return sample point and weights for integration, using
    a truncated logistic distribution as the base, and importance weights.
    """
    loc, scale, clip = t.tensor(loc, device=device), t.tensor(scale, device=device), t.tensor(clip, device=device)
    # IID samples from uniform, use inverse CDF to transform to target distribution
    ps = t.rand(npoints, device=device)
    ps = t.sigmoid(-clip) + (t.sigmoid(clip) - t.sigmoid(-clip)) * ps  # Scale quantiles to clip
    logsnr = loc + scale * t.logit(ps)  # Using quantile function for logistic distribution

    # importance weights
    weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
    return logsnr, weights
