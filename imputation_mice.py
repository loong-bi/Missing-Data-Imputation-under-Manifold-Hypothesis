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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor # Add this import
import ot

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
    k = torch.randint(0, dim_hd, (1,)).item()
    idx = torch.randperm(D)[:k]
    mask_combined[i, idx] = 0.0

# Initial baseline fill: Initialize missing values with train_mean
train_mean = torch.mean(xs, dim=0)
train_mean_broadcasted = train_mean.expand(N_combined, -1)
x_imputed_combined = torch.where(mask_combined == 1.0, xs_combined, train_mean_broadcasted)

# Convert the combined structures to numpy for standard sklearn processing
x_mice_np = x_imputed_combined.detach().cpu().numpy()
mask_np = mask_combined.detach().cpu().numpy()

max_mice_iters = 10
tolerance = 1e-4

print("Starting Pooled Linear MICE Imputation...")

for iteration in range(max_mice_iters):
    print(iteration)
    x_old = x_mice_np.copy()
    
    # Loop sequentially through each feature column
    for d in range(D):
        # Find which samples have this column observed vs missing across the POOLED dataset
        observed_rows = (mask_np[:, d] == 1.0)
        missing_rows = (mask_np[:, d] == 0.0)
        
        # If nothing is missing in this column, move to the next
        if not np.any(missing_rows):
            continue
            
        # All other columns are used as predictors
        predictor_cols = [col for col in range(D) if col != d]
        
        # Training includes clean 'xs' rows + currently available 'xs_test' rows
        X_train = x_mice_np[observed_rows][:, predictor_cols]
        y_train = x_mice_np[observed_rows][:, d]
        X_missing = x_mice_np[missing_rows][:, predictor_cols]
        
        # Standard Linear Regression engine
        #regressor = LinearRegression()
        regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=321)
        regressor.fit(X_train, y_train)
        
        # Predict and impute the missing coordinates
        x_mice_np[missing_rows, d] = regressor.predict(X_missing)

    # Calculate convergence based on how much the imputed values shifted
    delta = np.mean((x_mice_np - x_old) ** 2)
    
    if delta < tolerance:
        print(f"Standard MICE converged at iteration {iteration + 1}. Delta: {delta:.8f}")
        break
# else:
    # print(f"Standard MICE reached max iterations ({max_mice_iters}). Final Delta: {delta:.8f}")
    
# from sklearn.experimental import enable_iterative_imputer  # noqa
# from sklearn.impute import IterativeImputer
# from sklearn.ensemble import RandomForestRegressor

# print("Starting MissForest-style Iterative Imputation (train-only fit)...")

# # Split back using your original structure
# num_train = xs.size(0)

# x_train_np = x_mice_np[:num_train].copy()
# x_test_np = x_mice_np[num_train:].copy()

# # Fit ONLY on training data (no missing assumed there)
# imputer = IterativeImputer(
    # estimator=RandomForestRegressor(
        # n_estimators=100,
        # n_jobs=-1,
        # random_state=321
    # ),
    # max_iter=10,
    # random_state=321
# )

# imputer.fit(x_train_np)

# # Apply to test only (with mask applied)
# x_test_nan = x_test_np.copy()
# x_test_nan[mask_np[num_train:] == 0.0] = np.nan

# x_test_imputed = imputer.transform(x_test_nan)

# # Put back together (optional but usually needed for downstream code)
# x_mice_np = np.vstack([x_train_np, x_test_imputed])

# print("Imputation completed.")

# -----------------------------
# Post-Processing: Extract ONLY xs_test for Plotting
# -----------------------------
# Cast the entire array back to PyTorch
x_imputed_all = torch.from_numpy(x_mice_np).float().to(device)

# Slicing from num_train to the end extracts just the test set rows
x_seen_test = xs_combined[num_train:] * mask_combined[num_train:]
x_imputed_test = x_imputed_all[num_train:]

# real = x_seen_test.detach().cpu()
real = xs_test.detach().cpu()
fake = x_imputed_test.detach().cpu()

# -----------------------------
# Inverse transform before RMSE
# -----------------------------
real_orig = real 
fake_orig = fake

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

# from sklearn.neighbors import NearestNeighbors

# X = xs.numpy()
# Y_real = real.numpy()
# Y_fake = fake.numpy()

# # fit NN on training manifold
# nn = NearestNeighbors(n_neighbors=1).fit(X)

# # distance to manifold
# dist_real, _ = nn.kneighbors(Y_real)
# dist_fake, _ = nn.kneighbors(Y_fake)

# dist_real = dist_real.squeeze()
# dist_fake = dist_fake.squeeze()

# print("Mean NN distance (real → xs):", dist_real.mean())
# print("Mean NN distance (fake → xs):", dist_fake.mean())
# print("Gap (fake - real):", dist_fake.mean() - dist_real.mean())

# real = x_seen_test.detach().cpu()

# if dim_hd == 2:
    # fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # # compute shared limits
    # all_data = torch.cat([real, fake], dim=0)
    # xmin, ymin = all_data[:,0].min(), all_data[:,1].min()
    # xmax, ymax = all_data[:,0].max(), all_data[:,1].max()

    # # Real
    # axes[0].scatter(real[:,0], real[:,1], s=5, alpha=0.4, c="tab:blue")
    # axes[0].set_title("Missing")
    # axes[0].set_xlim(xmin, xmax)
    # axes[0].set_ylim(ymin, ymax)
    # axes[0].set_aspect('equal', adjustable='box')

    # # Fake
    # axes[1].scatter(fake[:,0], fake[:,1], s=5, alpha=0.4, c="tab:orange")
    # axes[1].set_title("Imputed")
    # axes[1].set_xlim(xmin, xmax)
    # axes[1].set_ylim(ymin, ymax)
    # axes[1].set_aspect('equal', adjustable='box')

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95)
    # plt.show()
    
if dim_hd == 2:

    all_data = torch.cat([real, fake], dim=0)
    xmin, ymin = all_data[:,0].min(), all_data[:,1].min()
    xmax, ymax = all_data[:,0].max(), all_data[:,1].max()

    # -------------------
    # Real (Missing)
    # -------------------
    plt.figure(figsize=(8,4))
    plt.scatter(real[:,0], real[:,1], s=5, alpha=0.4, c="tab:blue")
    #plt.title("Missing")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(
        os.path.join("C:/Users/z5281286/Desktop/project 05/Manifold_Mixture_VAEs", "missing.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -------------------
    # Fake (Imputed)
    # -------------------
    plt.figure(figsize=(8,4))
    plt.scatter(fake[:,0], fake[:,1], s=5, alpha=0.4, c="tab:orange")
    #plt.title("Imputed")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(
        os.path.join("C:/Users/z5281286/Desktop/project 05/Manifold_Mixture_VAEs", "imputed.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
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

else:
    print(f"Cannot visualize dim={dim_hd} directly")