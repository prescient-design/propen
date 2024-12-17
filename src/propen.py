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

from utils import * 

def reset_seeds(seed: int):
    """
    Set the random seed for Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MatchedDataset(data.Dataset):
    """
    A PyTorch Dataset for matched pairs of samples.
    
    Each entry consists of two samples (sample1 and sample2),
    created by splitting each row of data into two halves.
    """
    def __init__(self, dataset: np.ndarray, feat_size: int = 2):
        super().__init__()
        self.data = dataset
        self.feat_size = feat_size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        row = np.asarray(self.data[index])
        sample1 = row[:self.feat_size]
        sample2 = row[self.feat_size:]
        return torch.Tensor(sample1), torch.Tensor(sample2)


class MatchedAE(pl.LightningModule):
    """
    A matched AutoEncoder model using PyTorch Lightning.
    
    The model is designed to learn a transformation from an input sample (x_i)
    to a "matched" sample (x_c) through reconstruction loss.
    """
    def __init__(self, kde_to_eval, input_n=10, n_hidden=10, ae_type='propen'):
        super().__init__()
        self.input_n = input_n
        self.n_hidden = n_hidden
        self.ae_type = ae_type
        self.criterion = nn.MSELoss()
        self.kde = kde_to_eval

        # Encoder and Decoder definitions
        self.encoder = nn.Sequential(
            nn.Linear(self.input_n, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden // 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_hidden // 2, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.input_n),
        )

    def forward(self, x):
        encoded = self.encoder(x).to(self.device)
        decoded = self.decoder(encoded).to(self.device)
        return decoded

    def training_step(self, batch, batch_idx):
        x_i, x_c = batch
        reconstructed = self(x_i)

        if self.ae_type == 'propen':
            loss = self.criterion(reconstructed, x_c)
        elif self.ae_type == 'propen_mixup':
            loss = self.criterion(reconstructed, x_c) + self.criterion(reconstructed, x_i)
        return loss

    def validation_step(self, batch, batch_idx):
        x_i, x_c = batch
        reconstructed = self(x_i)

        if self.ae_type == 'propen':
            loss = self.criterion(reconstructed, x_c)
        elif self.ae_type == 'propen_mixup':
            loss = self.criterion(reconstructed, x_c) + self.criterion(reconstructed, x_i)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0)


def find_nearest_larger_y(y_i, y, y_th=0.001):
    """
    For a given value y_i, find indices of elements in y that are greater than y_i by at least y_th.
    
    Returns:
        indices: indices in y that meet the condition
        worse_grads: repeated array of y_i for indexing
        better_grads: corresponding entries from y that are larger than y_i.
    """
    y_diff = y - y_i
    indices = np.where(y_diff > y_th)[0]
    
    return indices


def match_dataset(x_2d_array, y_1d_array, y_th=0.1, x_th=1):
    """
    Given x_2d_array and y_1d_array, find pairs (x_i, x_c) such that:
    - x_c corresponds to a point with a larger y value (exceeding threshold y_th).
    - The Euclidean distance between x_i and x_c is less than x_th.

    Returns a NumPy array of matched pairs [x_i, x_c].
    """
    x = torch.Tensor(x_2d_array)
    y = torch.Tensor(np.asarray(y_1d_array))
    n_dim = x.shape[1]

    nearest_larger_ys = []
    for i in range(x.shape[0]):
        indices = find_nearest_larger_y(y[i], y, y_th)
        nearest_larger_ys.append(np.asarray(indices))

    xi = []
    xc = []
    for i in range(x.shape[0]):
        dists_x = [np.linalg.norm(x[i] - x[j]).mean() for j in nearest_larger_ys[i]]
        for di, dist_val in enumerate(dists_x):
            if dist_val <= x_th:
                xi.append(x[i])
                xc.append(x[nearest_larger_ys[i][di]])

    xi = torch.vstack(xi).reshape(-1, n_dim).numpy()
    xc = torch.vstack(xc).reshape(-1, n_dim).numpy()
    matched_pairs = np.hstack([xi, xc])
    return matched_pairs


def propen_forward_pass(matched_test, propen_model, device):
    """
    Forward pass of the Propen model on matched_test data.
    """
    x_i = torch.Tensor(matched_test).to(device)
    x_i_hat = propen_model(x_i)
    return x_i_hat.cpu().detach().numpy()


def n_steps_propen(input_xd, model, device, n_steps=10):
    """
    Iteratively apply the model n_steps times to input_xd and store the outputs.
    """
    lst_optimized_nd = []
    for _ in range(n_steps):
        forward_pass = propen_forward_pass(input_xd, model, device)
        lst_optimized_nd.append(forward_pass)
        input_xd = forward_pass
    return lst_optimized_nd


def evaluate_designs(lst_optimized, test_df, train_df, kde, compute_color_score):
    """
    Evaluate a list of optimization steps (lst_optimized) using various metrics:
    - Average improvement (AI)
    - Ratio improvement (RI)
    - Negative Log-Likelihood (NLL) from KDE
    - Uniqueness (UQ)
    - Novelty (NOV)

    If plot > 0, produce a visualization.
    """
    ais = []
    ris = []
    nlls = []
    uqs = []
    novs = []

    # Compute baseline test and train KDE scores
    kde_scores_test = compute_color_score(test_df[:, 1], test_df[:, 0]).reshape(-1)
    kde_scores_test = kde_scores_test.numpy()
    
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(lst_optimized)))

    for op_idx, decoded_op_2d in enumerate(lst_optimized):
        
        kde_scores_decoded_op = compute_color_score(decoded_op_2d[:, 1], decoded_op_2d[:, 0])

        nll = np.round(ood_score(kde, decoded_op_2d).sum(), 15)
        ai = np.round(average_improvement(kde_scores_test, kde_scores_decoded_op), 15)
        ri = np.round(ratio_improvement(kde_scores_test, kde_scores_decoded_op), 15)
        uq = np.round(uniqness(decoded_op_2d), 15)
        nov = np.round(novelty(decoded_op_2d, train_df), 15)

        ais.append(ai)
        ris.append(ri)
        nlls.append(nll)
        uqs.append(uq)
        novs.append(nov)

    return ais, ris, nlls, uqs, novs


