import torch.nn as nn
import torch
from utils.injective_generators import InjectiveDenseGenerator
from utils.left_invertible_vae import Mix_VAE
import utils.sampling
import numpy as np
import os
from utils.plot import plot_all,plot_generated
from utils.general import get_device
from utils.datasets import ToyDataset
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(321)
torch.manual_seed(321)
random.seed(321)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "sphere"
model_path = "models/mix_vae_sphere_4_gen_last.pt"

sig_ld=0.01
sig_hd=.01
num_generators=4
dim_hd=3
dim_ld=2
noise_level=0.01

n_data=1024*10
train_dataset=ToyDataset(model_name,n_data,noise_level=noise_level)
xs = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])

decoders = []

for _ in range(num_generators):
    decoders.append(
        InjectiveDenseGenerator(
            dim_hd,
            dim_ld,
            sig_hd,
            sig_ld,
            utils.sampling.get_spline_latent,
            latent_nf=(dim_ld > 1)
        ).to(device)
    )

model = Mix_VAE(decoders).to(device)

ckpt = torch.load(model_path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# p(c)
log_pi = model.get_log_decoder_weights()
pc = torch.exp(log_pi)

# obtain labels and latens
pcx = model.classify(xs)
cs = pcx.argmax(dim=1, keepdim=True)
cs = cs.view(-1)
# obtain mean first
zs = torch.empty(xs.size(0), dim_ld)
for c in cs.unique():
    mask = (cs == c)
    zs[mask] = model.decoders[c.item()].backward(xs[mask])
# if we want to be more stochastic
zs = zs + sig_ld * torch.randn_like(zs)

zs = zs.detach()
xs = xs.detach()
cs = cs.detach()

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)

# -----------------------------
# 1. Create test dataset and Combine with Training Data
# -----------------------------
n = 1024
test_dataset = ToyDataset(model_name, n, noise_level=noise_level)
xs_test = torch.stack([
    test_dataset[i][0] for i in range(len(test_dataset))
]).to(device)

# Ensure xs is on the correct device
xs = xs.to(device)
num_train = xs.size(0)
num_test = xs_test.size(0)

# Concatenate training data and test data along the batch dimension
xs_combined = torch.cat([xs, xs_test], dim=0)
N_combined, D = xs_combined.shape

# Build a combined mask: 
# - Training data rows are fully observed (all 1.0)
# - Test data rows are randomly masked
mask_combined = torch.ones_like(xs_combined).to(device)

for i in range(num_train, N_combined):
    k = torch.randint(1, dim_hd, (1,)).item()
    idx = torch.randperm(D)[:k]
    mask_combined[i, idx] = 0.0

# -----------------------------
# PCA Imputation Algorithm
# -----------------------------

# Step 1: Initialize missing values with train_mean across the combined matrix
train_mean = torch.mean(xs, dim=0)
train_mean_broadcasted = train_mean.expand(N_combined, -1)
x_imputed = torch.where(mask_combined == 1.0, xs_combined, train_mean_broadcasted)

# Hyperparameters for the PCA loop
max_iter = 10000
tolerance = 1e-6
prev_error = float("inf")

# Determine number of components to retain
n_components = max(1, D - 1)

for iteration in range(max_iter):

    # Step 2a: Standardise the data
    mean = torch.mean(x_imputed, dim=0)
    std = torch.std(x_imputed, dim=0, unbiased=True) + 1e-12
    x_scaled = (x_imputed - mean) / std
    
    # Step 2b & 2c: SVD to find principal components and eigenvalues
    U, S, Vh = torch.linalg.svd(x_scaled, full_matrices=False)
    
    # Step 3a: Reconstruct the data using the retained principal components
    U_retained = U[:, :n_components]
    S_retained = torch.diag(S[:n_components])
    Vh_retained = Vh[:n_components, :]
    
    x_scaled_reconstructed = U_retained @ S_retained @ Vh_retained
    
    # Step 3b: Convert back to original data space
    x_reconstructed = (x_scaled_reconstructed * std) + mean
    
    # Step 5: Compute reconstruction error of the OBSERVED values only (Train + unmasked Test)
    observed_diff = (x_reconstructed - xs_combined) * mask_combined
    reconstruction_error = torch.mean(observed_diff ** 2).item()
    
    # Step 4: Replace missing values with reconstructed values, keep observed unchanged
    x_imputed = torch.where(mask_combined == 1.0, xs_combined, x_reconstructed)
    
    # Step 6: Check convergence condition
    if abs(prev_error - reconstruction_error) < tolerance:
        print(f"PCA Imputation converged at iteration {iteration}. Error: {reconstruction_error:.6f}")
        break
        
    prev_error = reconstruction_error
else:
    print(f"PCA Imputation reached max iterations ({max_iter}). Final Error: {prev_error:.6f}")


# -----------------------------
# Post-Processing: Extract ONLY xs_test for Plotting
# -----------------------------
# Slicing from num_train to the end extracts just the test set rows
x_seen_test = xs_combined[num_train:] * mask_combined[num_train:]
x_imputed_test = x_imputed[num_train:]

real = xs_test.detach().cpu()
fake = x_imputed_test.detach().cpu()

import ot
# -----------------------------
# Inverse transform before RMSE
# -----------------------------
real_orig = real 
#torch.tensor(
    # scaler.inverse_transform(real.numpy()), dtype=torch.float32
# )
fake_orig = fake
# torch.tensor(
    # scaler.inverse_transform(fake.numpy()), dtype=torch.float32
# )

missing_mask = (mask_combined[num_train:] == 0).cpu()  # True where values were missing

# Squared error only at missing positions (in original scale)
se = (real_orig - fake_orig) ** 2
se_missing = se * missing_mask

n_missing_per_sample = missing_mask.sum(dim=1).float()
mse_per_sample = se_missing.sum(dim=1) / (n_missing_per_sample + 1e-12)
mse_total = se_missing.sum() / (missing_mask.sum().float() + 1e-12)
rmse_total = mse_total.sqrt()

print(f"RMSE over missing components: {rmse_total.item():.4f}")

X = real.numpy()
Y = fake.numpy()

n = X.shape[0]
m = Y.shape[0]

a = np.ones(n) / n
b = np.ones(m) / m

M = ot.dist(X, Y, metric='euclidean')

W = ot.emd2(a, b, M)

print("Wasserstein =", np.sqrt(W))

if dim_hd == 2:
    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # compute shared limits
    all_data = torch.cat([real, fake], dim=0)
    xmin, ymin = all_data[:,0].min(), all_data[:,1].min()
    xmax, ymax = all_data[:,0].max(), all_data[:,1].max()

    # Real
    axes[0].scatter(real[:,0], real[:,1], s=5, alpha=0.4, c="tab:blue")
    axes[0].set_title("Missing")
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    axes[0].set_aspect('equal', adjustable='box')

    # Fake
    axes[1].scatter(fake[:,0], fake[:,1], s=5, alpha=0.4, c="tab:orange")
    axes[1].set_title("Imputed")
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)
    axes[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# elif dim_hd == 3:
    # fig = plt.figure(figsize=(12,6))

    # all_data = torch.cat([real, fake], dim=0)
    # xmin, ymin, zmin = all_data[:,0].min(), all_data[:,1].min(), all_data[:,2].min()
    # xmax, ymax, zmax = all_data[:,0].max(), all_data[:,1].max(), all_data[:,2].max()

    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.scatter(real[:,0], real[:,1], real[:,2], s=5, alpha=0.4, c="tab:blue")
    # ax1.set_title("Missing")
    # ax1.set_xlim(xmin, xmax)
    # ax1.set_ylim(ymin, ymax)
    # ax1.set_zlim(zmin, zmax)
    # ax1.set_box_aspect([1,1,1])

    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.scatter(fake[:,0], fake[:,1], fake[:,2], s=5, alpha=0.4, c="tab:orange")
    # ax2.set_title("Imputed")
    # ax2.set_xlim(xmin, xmax)
    # ax2.set_ylim(ymin, ymax)
    # ax2.set_zlim(zmin, zmax)
    # ax2.set_box_aspect([1,1,1])

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95)
    # plt.show()


elif dim_hd == 3:

    all_data = torch.cat([real, fake], dim=0)
    xmin, ymin, zmin = all_data[:,0].min(), all_data[:,1].min(), all_data[:,2].min()
    xmax, ymax, zmax = all_data[:,0].max(), all_data[:,1].max(), all_data[:,2].max()
    
    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(real[:,0], real[:,1], real[:,2], s=5, alpha=0.4, c="tab:blue")

    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.set_zlim(zmin, zmax)
    # ax.grid(False)

    # ax.set_box_aspect([1, 1, 1])  # works for 3D axes

    # plt.savefig(
        # os.path.join(
            # "C:/Users/z5281286/Desktop/project 05/Manifold_Mixture_VAEs",
            # "missing.png"
        # ),
        # dpi=300,
        # bbox_inches="tight"
    # )

    # plt.close(fig)
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(fake[:,0], fake[:,1], fake[:,2], s=5, alpha=0.4, c="tab:orange")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.grid(False)

    ax.set_box_aspect([1, 1, 1])  # works for 3D axes

    plt.savefig(
        os.path.join(
            "C:/Users/z5281286/Desktop/project 05/Manifold_Mixture_VAEs",
            "imputed.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

else:
    print(f"Cannot visualize dim={dim_hd} directly")