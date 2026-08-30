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
import ot
from sklearn.ensemble import RandomForestRegressor

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

# method 03: latent diffusion model with one chart
# noise schedule for continuous and discrete cases
T = 500
beta = torch.linspace(1e-4, 0.02, T)
K = num_generators
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
Q = []
for b in beta:
    q = (1 - b) * torch.eye(K) + b * torch.ones(K, K) / K # with probability 1 - b keep the label, otherwise sample uniformly
    Q.append(q)
    
Q = torch.stack(Q) # [T, K, K]

Q_cum = torch.zeros_like(Q)
Q_cum[0] = Q[0] # for c_0 to c_1
for t in range(1, T):
    Q_cum[t] = Q_cum[t-1] @ Q[t]

# a function for accelerating tensor operation
def extract(a, t, x_shape):
    out = a.gather(0, t) # select a values at t indexes
    return out.view(-1, *([1] * (len(x_shape) - 1))) 

# forward diffusion process for a point
def q_sample(x_0, noise, c_0, t): # x_0 is batch * shape, t is batch * 1
    
    a_bar = extract(alpha_bar, t, x_0.shape) # for each t value, extract the corresponding alpha_bar, then make it a tensor the same shape as x_0
    x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * noise
    
    probs = Q_cum[t][torch.arange(c_0.shape[0]), c_0] # each t gives us a matrix and then we pick the row corresponding to the value and arrange them in the same order as in batch
    c_t = torch.multinomial(probs, 1).squeeze(-1) # squeeze means remove the last dimension if it is size 1
    
    return x_t, c_t
    
# we might need a better cross-attention design
class JointDiffusion(nn.Module):

    def __init__(self, dim, n_classes=2, hidden=256):
        super().__init__()
        
        # time and class embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        self.class_emb = nn.Embedding(n_classes, hidden)

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(dim + 2 * hidden, hidden), # data, class and time embedding as input
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

        # head for x (continuous)
        self.x_head = nn.Linear(hidden, dim)

        # head for c (discrete logits)
        self.c_head = nn.Linear(hidden, n_classes)

    def forward(self, x, c, t):

        t = t.float().unsqueeze(-1) / T
        t_emb = self.time_mlp(t)
        c_emb = self.class_emb(c)

        h = torch.cat([x, t_emb, c_emb], dim=1)
        h = self.trunk(h)

        x_out = self.x_head(h)        # predicts noise or x_0
        c_logits = self.c_head(h)     # predicts logits over classes

        return x_out, c_logits


model_dfzc = JointDiffusion(dim_ld, K).to(device)
opt = torch.optim.Adam(model_dfzc.parameters(), lr=1e-3)
steps = 10000 # 1 epoch
batch_size = 256
        
for step in range(steps):
    idx = torch.randint(0, zs.shape[0], (batch_size,), device=device)
    z_0 = zs[idx]
    z_0 = z_0.to(device)
    c_0 = cs[idx]
    c_0 = c_0.to(device)

    # diffusion timestep
    t = torch.randint(0, T, (batch_size,), device=device)

    # forward process
    noise = torch.randn_like(z_0)
    z_t, c_t = q_sample(z_0, noise, c_0, t)
    z_noise, c_logits = model_dfzc(z_t, c_t, t)
    
    loss = F.mse_loss(z_noise, noise) + F.cross_entropy(c_logits, c_0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 500 == 0:
        print(f"step {step}, loss {loss.item():.4f}")
        
@torch.no_grad()
def sample_04(model, n=256, dim=2, K=1):
    z = torch.randn(n, dim).to(device)
    c = torch.randint(0, K, (n,), device=device)
    
    # sample from p(x_0, c_0|x_t, c_t) first
    for t in reversed(range(T)):
        t = torch.full((n,), t, device=device, dtype=torch.long)
        z_out, c_logits = model(z, c, t)
        
        t_prev = torch.clamp(t - 1, min=0)
        alpha_t = extract(alpha, t, z.shape)
        alpha_t_prev = extract(alpha, t_prev, z.shape)
        alpha_bar_t = extract(alpha_bar, t, z.shape)
        alpha_bar_prev = extract(alpha_bar, t_prev, z.shape)
        
        z_0 = (z - torch.sqrt(1 - alpha_bar_t) * z_out) / torch.sqrt(alpha_bar_t)
        #c_0 = torch.argmax(c_logits, dim=-1)
        c_0 = torch.multinomial(torch.softmax(c_logits, dim=-1), 1).squeeze(-1)
        
        if t[0] > 0:
            # sample from p(z_{t - 1}|z_t, z_0)
            mean = (
                torch.sqrt(alpha_t) * (1 - alpha_bar_prev) * z
                + torch.sqrt(alpha_bar_prev) * (1 - alpha_t) * z_0
            ) / (1 - alpha_bar_t)

            var = ((1 - alpha_bar_prev) * (1 - alpha_t)) / (1 - alpha_bar_t)
            std = torch.sqrt(var)

            z = mean + std * torch.randn_like(z)
            
            # sample from p(c_{t - 1}|c_t, c_0)
            probs = torch.zeros(n, K, device=device)

            for k in range(K):
                # Q_t(c_t | k)
                q_ct_given_k = Q[t[0], k][c]    # [n]
                # Q_{0->t-1}(k | c0)
                q_k_given_c0 = Q_cum[t[0] - 1][c_0][:, k]  # [n]
                probs[:, k] = q_ct_given_k * q_k_given_c0
            
            # normalize the probabilities
            probs = probs / probs.sum(dim=-1, keepdim=True)
            c = torch.multinomial(probs, 1).squeeze(-1)
            
    return z, c
    

# create missing dataset

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


# -----------------------------
# 2. Create missing mask
# -----------------------------
mask = torch.ones_like(xs_test)

for i in range(N):
    k = torch.randint(0, dim_hd, (1,)).item()  # 1 or 2 missing dims
    idx = torch.randperm(D)[:k]
    mask[i, idx] = 0.0

x_seen = xs_test * mask


# -----------------------------
# 3. Sample from joint prior p(z, c)
# -----------------------------
M = 10000
z_new, c_new = sample_04(model_dfzc, M, dim_ld, num_generators)

# -----------------------------
# 4. Decode prior samples
# -----------------------------
x_new = torch.zeros((M, D), device=z_new.device)

for k in range(K):
    mk = (c_new == k)
    if mk.sum() == 0:
        continue
    x_new[mk] = model.decoders[k](z_new[mk])


# -----------------------------
# 5. Likelihood weights
# -----------------------------
def compute_weights(x_samples, x_seen, mask, sigma):
    diff = (x_samples - x_seen) * mask
    dist2 = (diff ** 2).sum(dim=1)
    w = torch.exp(-dist2 / (2 * sigma**2))
    return w


# -----------------------------
# 6. SIR imputation per sample
# -----------------------------
x_imputed_list = []

for i in range(N):

    x_seen_i = x_seen[i].unsqueeze(0).repeat(M, 1)
    mask_i = mask[i].unsqueeze(0).repeat(M, 1)
    ws = compute_weights(x_new, x_seen_i, mask_i, sig_hd)
    ws = ws / (ws.sum() + 1e-12)
    
    if ws.sum() <= 0:
        ws = torch.ones_like(ws) / ws.size(0)
    
    idx = torch.multinomial(ws, 1).squeeze(0)
    # idx = torch.argmax(ws)
    z_star = z_new[idx].unsqueeze(0)
    c_star = c_new[idx].item()
    x_full = model.decoders[c_star](z_star)
    x_full = x_full + sig_hd * torch.randn_like(x_full)
    
    # K_post = 10
    # samples = []
    # for _ in range(K_post):
        # idx = torch.multinomial(ws, 1).squeeze(0)
        # z_star = z_new[idx].unsqueeze(0)
        # c_star = c_new[idx].item()
        # x_full = model.decoders[c_star](z_star)
        # x_full = x_full + sig_hd * torch.randn_like(x_full)
        # samples.append(x_full[0])

    # samples = torch.stack(samples)
    # x_mean = samples.mean(dim=0)

    

    # fill missing entries
    x_imputed = x_seen[i].clone()
    x_imputed[mask[i] == 0] = x_full[0][mask[i] == 0]
    # x_imputed[mask[i] == 0] = x_mean[mask[i] == 0]

    x_imputed_list.append(x_imputed)


# -----------------------------
# 7. Final tensor
# -----------------------------
x_imputed = torch.stack(x_imputed_list)

##########################################################################
# Initial baseline fill: Initialize missing values with proposals
# Convert the combined structures to numpy for standard sklearn processing
# xs = xs.to(device)
# num_train = xs.size(0)
# num_test = xs_test.size(0)

# # Concatenate training data and test data along the batch dimension
# xs_combined = torch.cat([xs, xs_test], dim=0)
# N_combined, D = xs_combined.shape

# # Build a combined mask: 
# # - Training data rows are fully observed (all 1.0)
# # - Test data rows are randomly masked
# mask_combined = torch.ones_like(xs_combined).to(device)

# for i in range(num_train, N_combined):
    # k = torch.randint(0, dim_hd, (1,)).item()
    # idx = torch.randperm(D)[:k]
    # mask_combined[i, idx] = 0.0

# x_mice_init = xs_combined.clone()
# x_mice_init[num_train:] = x_imputed

# x_mice_np = x_mice_init.detach().cpu().numpy()
# mask_np = mask_combined.detach().cpu().numpy()

# max_mice_iters = 10
# tolerance = 1e-4

# print("Starting Pooled Linear MICE Imputation...")

# for iteration in range(max_mice_iters):
    # print(iteration)
    # x_old = x_mice_np.copy()
    
    # # Loop sequentially through each feature column
    # for d in range(D):
        # # Find which samples have this column observed vs missing across the POOLED dataset
        # observed_rows = (mask_np[:, d] == 1.0)
        # missing_rows = (mask_np[:, d] == 0.0)
        
        # # If nothing is missing in this column, move to the next
        # if not np.any(missing_rows):
            # continue
            
        # # All other columns are used as predictors
        # predictor_cols = [col for col in range(D) if col != d]
        
        # # Training includes clean 'xs' rows + currently available 'xs_test' rows
        # X_train = x_mice_np[observed_rows][:, predictor_cols]
        # y_train = x_mice_np[observed_rows][:, d]
        # X_missing = x_mice_np[missing_rows][:, predictor_cols]
        
        # # Standard Linear Regression engine
        # # regressor = LinearRegression()
        # regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=321)
        # regressor.fit(X_train, y_train)
        
        # # Predict and impute the missing coordinates
        # x_mice_np[missing_rows, d] = regressor.predict(X_missing)

    # # Calculate convergence based on how much the imputed values shifted
    # delta = np.mean((x_mice_np - x_old) ** 2)
    
    # if delta < tolerance:
        # print(f"Standard MICE converged at iteration {iteration + 1}. Delta: {delta:.8f}")
        # break
        
# x_imputed_all = torch.from_numpy(x_mice_np).float().to(device)

# # Slicing from num_train to the end extracts just the test set rows
# x_seen_test = xs_combined[num_train:] * mask_combined[num_train:]
# x_imputed_test = x_imputed_all[num_train:]

# # real = x_seen_test.detach().cpu()
# real = xs_test.detach().cpu()
# fake = x_imputed_test.detach().cpu()
###########################################################################


real = xs_test.detach().cpu()
fake = x_imputed.detach().cpu()


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


real = x_seen.detach().cpu()

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
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(real[:,0], real[:,1], real[:,2], s=5, alpha=0.4, c="tab:blue")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.grid(False)

    ax.set_box_aspect([1, 1, 1])  # works for 3D axes

    plt.savefig(
        os.path.join(
            "C:/Users/z5281286/Desktop/project 05/Manifold_Mixture_VAEs",
            "missing.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    
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