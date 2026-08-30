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
import torch.optim as optim
from gain import gain
import tensorflow as tf

tf.random.set_seed(321)
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
# 1. Create test dataset
# -----------------------------
n = 1024
test_dataset = ToyDataset(model_name, n, noise_level=noise_level)
xs_test = torch.stack([
    test_dataset[i][0] for i in range(len(test_dataset))
])

N, D = xs_test.shape

# Calculate the mean across the batch dimension (dim=0) from the training data
train_mean = torch.mean(xs, dim=0)

mask = torch.ones_like(xs_test)
for i in range(N):
    k = torch.randint(1, dim_hd, (1,)).item()
    idx = torch.randperm(D)[:k]
    mask[i, idx] = 0.0
    
x_seen_test = xs_test * mask + train_mean * (1.0 - mask)

# All-ones mask for the complete training data
mask_train = torch.ones_like(xs)
# Combine datasets and masks along the batch dimension
combined_data = torch.cat([xs, x_seen_test], dim=0)
combined_mask = torch.cat([mask_train, mask], dim=0)


# Assuming data is non-negative, but you need [0, 1] range
data_min = combined_data.min()
data_max = combined_data.max()
combined_data_norm = (combined_data - data_min) / (data_max - data_min)

# Now proceed with your masking
gain_data_np = combined_data_norm.detach().cpu().numpy()
gain_data_np[combined_mask == 0] = np.nan

gain_parameters = {
    'batch_size': 128,
    'hint_rate': 0.5,
    'alpha': 100,
    'iterations': 10000  # Adjust based on your time constraints
}

# Run the official GitHub training loop implementation
print("Starting GAIN training loop via TensorFlow...")
imputed_data_np = gain(gain_data_np, gain_parameters)
print("Imputation completed successfully.")

# -------------------------------------------------------------
# 5. Extract, Reconvert, and DENORMALIZE
# -------------------------------------------------------------
num_train_samples = xs.shape[0]
test_imputed_np = imputed_data_np[num_train_samples:]

# Convert to tensor
x_imputed = torch.tensor(test_imputed_np, dtype=torch.float32)

# --- DENORMALIZE ---
# Reverse: x = x_norm * (max - min) + min
x_imputed = x_imputed * (data_max - data_min) + data_min

# Now move to device for downstream logic
x_seen = x_seen_test.to(device)
x_imputed = x_imputed.to(device)
print(x_seen_test)
print(x_imputed)
# -------------------------------------------------------------
# Your Plotting Logic Continues Below...
# -------------------------------------------------------------
real = xs_test.detach().cpu()
fake = x_imputed.detach().cpu()


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

missing_mask = (mask == 0).cpu()  # True where values were missing

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