import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset


def reset_seeds(seed: int):
    """
    Set the random seed for Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def colored_gaussians(num_gaussians=8, points_per_gaussian=200, spread=2.0, rng=1):
    """
    Generates Gaussian clusters arranged in a circular pattern.

    Returns:
        dataset: (N, 2) NumPy array of points
        colors: (N,) NumPy array of angles (used as color values).
    """
    if rng is None:
        rng = np.random.RandomState()
    else:
        np.random.seed(rng)

    x_points, y_points, colors = [], [], []
    angles = np.linspace(0, 2 * np.pi, num_gaussians, endpoint=False)

    for angle in angles:
        mean = [spread * np.cos(angle), spread * np.sin(angle)]
        covariance = [[0.1, 0], [0, 0.1]]
        gaussian_points = np.random.multivariate_normal(mean, covariance, points_per_gaussian)

        x, y = gaussian_points[:, 0], gaussian_points[:, 1]
        x_points.extend(x)
        y_points.extend(y)
        colors.extend([angle] * points_per_gaussian)

    dataset = np.hstack([np.array(x_points).reshape(-1, 1), np.array(y_points).reshape(-1, 1)])
    return dataset, np.array(colors)


def compute_color(x, y):
    """
    Compute a normalized color value between 0 and 1 based on the angle of the point (x,y).
    """
    angle = np.arctan2(y, x)  # Angle in radians
    color_value = (angle + np.pi) / (2 * np.pi)
    return color_value


def create_colored_dataset(n_size=150, shape='8gaussians', seed=1):
    """
    Create a dataset of colored Gaussian samples, normalize colors, and fit a KernelDensity estimator.

    Returns:
        df_3d: DataFrame with columns ['x', 'y', 'z']
        compute_color: function to compute color
        kde: Fitted KernelDensity estimator
    """
    dataset, colors = colored_gaussians(num_gaussians=80,
                                        points_per_gaussian=n_size // 80,
                                        spread=2.0,
                                        rng=seed)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(colors.reshape(-1, 1)).astype('float32')

    df_3d = pd.DataFrame(np.hstack([
        dataset[:, 0].reshape(-1, 1),
        dataset[:, 1].reshape(-1, 1),
        normalized_data.reshape(-1, 1)
    ]), columns=['x', 'y', 'z'])

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(dataset)
    return df_3d, compute_color, kde


# Metrics functions
def average_improvement(true_values, sampled_values):
    """
    Compute the average improvement as the mean difference of sampled_values over true_values,
    considering only entries where sampled_values > true_values.
    """
    which_ones = np.where(sampled_values > true_values)[0]
    differences = (sampled_values - true_values)
    if which_ones.shape[0] > 0:
        return (differences[which_ones[0]]).sum() / which_ones.shape[0]
    else:
        return 0


def ratio_improvement(true_values, sampled_values):
    """
    Compute the percentage of samples where sampled_values > true_values.
    """
    how_many = np.where(sampled_values > true_values)[0].shape[0]
    return how_many / true_values.shape[0] * 100


def ood_score(kde, sampled_values):
    """
    Compute the out-of-distribution (negative log-likelihood) score for sampled_values using the given KDE.
    """
    return kde.score_samples(sampled_values)


def uniqness(sampled_values):
    """
    Compute the uniqueness percentage of sampled_values.
    """
    unique_count = np.unique(sampled_values, axis=0).shape[0]
    return (unique_count / sampled_values.shape[0]) * 100


def novelty(sampled_values, train_data):
    """
    Compute how many of the sampled_values are not present in the train_data, as a percentage.
    """
    new_samples = np.asarray(sampled_values)
    train_samples = np.asarray(train_data)
    copies = [(new_samples[x, :] in train_samples) for x in np.arange(new_samples.shape[0])]
    return len(copies) / sampled_values.shape[0] * 100


