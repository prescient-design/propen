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

from propen import *
from utils import *


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser('Propen toy experiments')
    parser.add_argument('--dataset_shape', type=str, default="pinwheel")
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--n_hidden', type=int, default=30)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--pad_dim', type=int, default=2)
    parser.add_argument('--n_hidden_units', type=int, default=30)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--test_train_ratio', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--y_th', type=float, default=0.1)
    parser.add_argument('--x_th', type=float, default=2)
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--step_size', type=float, default=1e-2)
    parser.add_argument('--results_dir', type=str, default="./output_dir/")
    parser.add_argument('--propen_type', type=str, default="propen_mixup")
    parser.add_argument('--output_dir', type=str, default="output_dir")
    parser.add_argument('--wandb', type=int, default=0)
    args = parser.parse_args()

    reset_seeds(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    seed = args.seed
    input_dim = args.pad_dim
    n_samples = args.n_samples
    batch_size = args.bs
    test_train_ratio = args.test_train_ratio
    num_workers = 5
    n_hidden = args.n_hidden
    n_steps = args.n_steps
    n_epochs = args.n_epochs
    propen_type = args.propen_type
    
    reset_seeds(seed)

    df_3d, compute_color_score, kde_3d = create_colored_dataset(n_size=n_samples, shape='8gaussians', seed=seed)
    df_3d['z'] = compute_color(df_3d['y'], df_3d['x'])

    df_3d_sorted = df_3d.sort_values('z', ascending=True)
    df_3d_sorted_bottom = df_3d_sorted.head(70)
    
    plt.figure(figsize=(6, 5), dpi=100)
    plt.scatter(df_3d_sorted_bottom['x'], df_3d_sorted_bottom['y'], c=df_3d_sorted_bottom['z'])
    plt.title('Train set')
    plt.colorbar()
    plt.savefig('train_data.png', dpi=180)
    plt.show()

    # Create a larger dataset for valid KDE
    _, _, kde_3d_large = create_colored_dataset(n_size=500)

    # Match dataset
    matched_pairs = match_dataset(np.asarray(df_3d_sorted_bottom[['x', 'y']]),
                                  np.asarray(df_3d_sorted_bottom['z']),
                                  y_th=args.y_th, x_th=args.x_th)

    matched_train, matched_test = train_test_split(matched_pairs, test_size=test_train_ratio)
    matched_test = matched_test[:, :input_dim]
    matched_test = np.unique(matched_test, axis=0)
    matched_test_tensor = torch.Tensor(matched_test[:, :input_dim])

    matched_train_2d = matched_train[:, :input_dim]
    matched_train_comp_2d = matched_train[:, input_dim:]

    plt.figure(figsize=(6, 5), dpi=100)
    plt.scatter(df_3d_sorted_bottom['x'], df_3d_sorted_bottom['y'], c=df_3d_sorted_bottom['z'])

    for i in range(matched_train_2d.shape[0]):
        plt.plot([matched_train_2d[i, 0], matched_train_comp_2d[i, 0]],
                 [matched_train_2d[i, 1], matched_train_comp_2d[i, 1]],
                 color='black', lw=0.5, alpha=0.5)

    plt.colorbar()
    plt.title('Matched pairs')
    plt.savefig('train_data_matched.png', dpi=180)
    plt.show()

    print('Train unique size:', np.unique(matched_train[:, :input_dim], axis=0).shape[0])
    print('Train pairs size:', matched_train.shape)
    print('Test pairs size:', matched_test_tensor.shape)

    seed = 1
    pl.seed_everything(seed)
    reset_seeds(seed)

    # Data loaders
    train_loader = DataLoader(
        MatchedDataset(matched_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        MatchedDataset(matched_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    model = MatchedAE(kde_3d, input_n=(input_dim),
                      n_hidden=n_hidden,
                      ae_type=propen_type)

    model.to(device)
    model.train()
    print(model)

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=n_epochs,
        logger=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=1,
        val_check_interval=1,
        check_val_every_n_epoch=10,
        enable_progress_bar=True,
    )

    # Train the model
    print("# Training commences ##########")
    trainer.fit(model, train_loader, valid_loader)
    print("# Training done ###############")

    # Optimization steps
    model.to(device)
    lst_n_propen_designs = n_steps_propen(matched_test_tensor, model, device, n_steps=n_steps)

    ai, ri, ood, uq, nov = evaluate_designs(
        lst_n_propen_designs,
        matched_test_tensor[:, :2], 
        matched_train[:, :2], 
        kde_3d,
        compute_color_score
    )

    # Print initial and final results
    print(f"Initial: AI={ai[0]} RI={ri[0]} NLL={ood[0]} UQ={uq[0]} NOV={nov[0]}")
    print(f"Final:   AI={ai[-1]} RI={ri[-1]} NLL={ood[-1]} UQ={uq[-1]} NOV={nov[-1]}")

    plt.figure(figsize=(6, 5), dpi=100)
    plt.scatter(df_3d_sorted_bottom['x'], df_3d_sorted_bottom['y'], c=df_3d_sorted_bottom['z'], alpha=0.3)
    plt.colorbar()

    # Visualize the optimization process for the first sample
    new_pairs = []
    for idx in range(matched_test_tensor.shape[0])[:1]:
        for op in range(n_steps):
            s1 = matched_test_tensor[idx, :]
            s2_idx = lst_n_propen_designs[op][idx].T
            new_pair = np.hstack([s1, s2_idx])
            new_pairs.append(new_pair)

            plt.plot(new_pair[2], new_pair[3], 'o', c='black', alpha=0.5)
            plt.plot(new_pair[0], new_pair[1], 'x', c='black', ms=10)

    plt.plot(new_pair[0], new_pair[1], 'x', c='black', label='starting point')
    plt.plot(new_pair[2], new_pair[3], 'o', c='black', alpha=0.1, label='propen design')
    plt.legend()

    new_pairs = np.asarray(new_pairs)
    plt.show()
    plt.savefig('propen_demo_results.png')
