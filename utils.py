# Utilities, including visualization and integration routine
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

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

def score(dist, x):
    """Compute score function (gradient of log p(x)) for a Pytorch distribution, dist, at points, x."""
    x = t.autograd.Variable(x, requires_grad=True)
    return t.autograd.grad(dist.log_prob(x).sum(), [x])[0]

def gauss_contour(cov, r, npoints=100):
    """Output a contour of a 2D Gaussian distribution."""
    theta = np.linspace(0, 2 * np.pi, npoints)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xy = np.stack((x, y), axis=1)
    mat = sqrtm(cov)
    return np.einsum('ij,kj->ki', mat, xy)

#### VISUALIZATION UTILITIES ####

def plot_mse(logsnrs, mses, mmse_g):
    fig, ax = plt.subplots(1, 1)
    ax.plot(logsnrs, mses, label="MSE")
    ax.plot(logsnrs, mmse_g, label='MMSE Gaussian')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\\alpha$)')
    ax.legend()
    return fig

def plot_density(x, logp1, logp2, labels):
    fig, axs = plt.subplots(1, len(labels), sharex=True, sharey=False, figsize=(6 * len(labels), 6))
    for i, ax in enumerate(axs):
        ax.set_title(labels[i])
        ax.plot(x, logp1[i], label='Diff. estimate')
        ax.plot(x, logp2[i], label='True')
        ax.set_ylabel('$\log p_{\\alpha}(\\vec x + \eta \cdot \\vec v)$')
        ax.set_xlabel('$\eta$')
        ax.legend()
    return fig

def vector_field(x_grid, grads, labels, scale=None):
    """Plot vector field for some function, grad, on Axis, ax."""
    xv, yv = x_grid[:,0], x_grid[:,1]
    fig, axs = plt.subplots(1, len(grads), sharex=True, sharey=True, figsize=(6 * len(grads), 6))
    for i, ax in enumerate(axs):
        if i > 0:
            scale = q.scale
            print("scale", scale)
        ax.set_title(labels[i])
        # grads = grad_fs[i](x_grid)
        q = ax.quiver(xv, yv, grads[i][:,0], grads[i][:,1], scale=scale, scale_units='inches')
        q._init()
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
    return fig

def vector_field2(x_grid, grad1, grad2, labels1, labels2, scale=None):
    """Plot vector fields on two rows, with various values for columns."""
    fig, axs = plt.subplots(2, len(grad2), sharex=True, sharey=True, figsize=(6 * len(grad2), 6*2))
    for i in range(len(grad1)):
        ax = axs[0, i]
        if i > 0:
            scale = q.scale
        ax.set_title(labels1[i])
        q = ax.quiver(x_grid[:,0], x_grid[:,1], grad1[i][:,0] - grad2[i][:,0], grad1[i][:,1] - grad2[i][:,1], scale=scale, scale_units='inches')
        q._init()
        print('scale', q.scale)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

    for i in range(len(grad2)):
        ax = axs[1, i]
        scale = None  # auto-scale each time
        ax.set_title(labels2[i])
        q = ax.quiver(x_grid[:,0], x_grid[:,1], grad2[i][:,0], grad2[i][:,1], scale=scale, scale_units='inches')
        q._init()
        print('scale row 2', q.scale)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

    fig.delaxes(axs[0, -1])
    return fig

def vector_field3(x_grid, grad1, grad2, labels1, labels2, scale=None):
    """Plot vector fields on two rows, with various values for columns."""
    fig, axs = plt.subplots(1, len(grad1), sharex=True, sharey=True, figsize=(6 * len(grad1), 6))
    for i in range(len(grad1)):
        ax = axs[i]
        ax.set_title(labels1[i])
        q = ax.quiver(x_grid[:,0], x_grid[:,1], grad1[i][:,0], grad1[i][:,1], scale=scale, scale_units='inches', label='Est.')
        q._init()
        ax.quiver(x_grid[:, 0], x_grid[:, 1], grad2[i][:, 0], grad2[i][:, 1], scale=q.scale, scale_units='inches', label='True', color='r', alpha=0.5)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.legend()

    return fig