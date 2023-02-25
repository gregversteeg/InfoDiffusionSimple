"""
Nice 2-d examples from: https://github.com/tanelp/tiny-diffusion
Standardizing the data, scale to have unit variance, as we do here. helps with interpretability,
as discussed in "Information-theoretic diffusion". The gap between MMSE and gaussian MMSE should be non-negative
and log likelihood can be interpreted as a stochastic lower bound on the true log likelihood.
"""
import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


def scg(n=20000):
    """Strongly correlated Gaussian"""
    # Load synthetic dataset, visualize and make dataloader
    cov = torch.tensor([[1., -0.4950 / 0.5050], [-0.4950 / 0.5050, 1.]])
    mu = torch.zeros(2)
    q_d = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
    # TODO: Can use "transformed distributions" to do more complex distributions - pyro or flowtorch?
    print("True entropy of this Gaussian distribution is:", q_d.entropy())
    return TensorDataset(q_d.sample((n,)))


def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    # This ends up looking like a square because we standardize the data
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
    df = pd.read_csv("assets/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def get_dataset(name, n=8000):
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    elif name == "scg":
        return scg(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")