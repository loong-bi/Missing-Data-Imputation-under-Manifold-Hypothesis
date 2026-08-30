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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import ot
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_index", type=int, default=1)
args = parser.parse_args()

np.random.seed(321)
torch.manual_seed(321)
random.seed(321)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_index = args.model_index
model_names = ["superconductivity", "energydata", "finance", "powerplant", "wine", "sensorless", "facial_expressions"]
chart_nums = [6, 6, 6, 2, 1, 6, 6]
dim_hds = [78, 24, 50, 4, 11, 48, 200]
dim_lds = [2, 3, 4, 3, 2, 8, 3]
n_datas = [1024 * 9, 1024 * 8, 1024 * 20, 1024 * 3, 1024 * 1, 1024 * 27, 1024 * 12]
n_test_datas = [1024, 1024, 1024, 1024, 1024, 1024, 1024]

sig_ld=0.01
sig_hd=.01
noise_level=0.01
num_generators=chart_nums[model_index]
dim_hd=dim_hds[model_index]
dim_ld=dim_lds[model_index]
n_data=n_datas[model_index]
n_test_data=n_test_datas[model_index]
model_name = model_names[model_index]
model_path = "models/mix_vae_" + model_name + "_" + str(num_generators) + "_gen_last.pt"

file = model_name + "/train_50_original.csv"
df = pd.read_csv(file)
data = df.values
scaler = StandardScaler()
data = scaler.fit_transform(data)
transform = transforms.ToTensor()
all_idx = np.random.choice(len(data), size=n_data + n_test_data, replace=False)
train_idx = all_idx[:n_data]
train = data[train_idx]
# convert NumPy arrays to PyTorch tensors first
train_ones = torch.tensor(train, dtype=torch.float32)
train_dataset = train_ones.view(len(train_ones), -1).float()
train_dataset = ToyDataset(data=train_dataset)
xs = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])

files = [
    model_name + "/test_50_true.csv",
    model_name + "/test_" + model_name + "_MCAR_10percent.csv",
    model_name + "/test_" + model_name + "_MCAR_20percent.csv",
    model_name + "/test_" + model_name + "_MCAR_30percent.csv",
    model_name + "/test_" + model_name + "_MCAR_40percent.csv",
    model_name + "/test_" + model_name + "_MCAR_50percent.csv",
    model_name + "/test_" + model_name + "_MCAR_60percent.csv",
    model_name + "/test_" + model_name + "_MCAR_70percent.csv",
    model_name + "/test_" + model_name + "_MCAR_80percent.csv",
    model_name + "/test_" + model_name + "_MCAR_90percent.csv",
    
    model_name + "/test_" + model_name + "_MAR_10percent.csv",
    model_name + "/test_" + model_name + "_MAR_20percent.csv",
    model_name + "/test_" + model_name + "_MAR_30percent.csv",
    model_name + "/test_" + model_name + "_MAR_40percent.csv",
    model_name + "/test_" + model_name + "_MAR_50percent.csv",
    model_name + "/test_" + model_name + "_MAR_60percent.csv",
    model_name + "/test_" + model_name + "_MAR_70percent.csv",
    model_name + "/test_" + model_name + "_MAR_80percent.csv",
    model_name + "/test_" + model_name + "_MAR_90percent.csv",
    
    model_name + "/test_" + model_name + "_MNAR_10percent.csv",
    model_name + "/test_" + model_name + "_MNAR_20percent.csv",
    model_name + "/test_" + model_name + "_MNAR_30percent.csv",
    model_name + "/test_" + model_name + "_MNAR_40percent.csv",
    model_name + "/test_" + model_name + "_MNAR_50percent.csv",
    model_name + "/test_" + model_name + "_MNAR_60percent.csv",
    model_name + "/test_" + model_name + "_MNAR_70percent.csv",
    model_name + "/test_" + model_name + "_MNAR_80percent.csv",
    model_name + "/test_" + model_name + "_MNAR_90percent.csv",
]

# -----------------------------
# test datasets
# -----------------------------
file = files[0]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_complete = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[1]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_10 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[2]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_20 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[3]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_30 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[4]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_40 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[5]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_50 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[6]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_60 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[7]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_70 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[8]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_80 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[9]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mcar_90 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[10]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_10 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[11]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_20 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[12]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_30 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[13]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_40 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[14]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_50 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[15]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_60 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[16]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_70 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[17]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_80 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[18]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mar_90 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[19]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_10 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])


file = files[20]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_20 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])


file = files[21]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_30 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[22]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_40 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[23]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_50 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

file = files[24]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_60 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])


file = files[25]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_70 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])


file = files[26]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_80 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])


file = files[27]
_df = pd.read_csv(file)
_data = _df.values
_data = scaler.transform(_data)
_test_ones  = torch.tensor(_data,  dtype=torch.float32)
_test_dataset  = _test_ones.view(len(_test_ones), -1).float()
_test_dataset = ToyDataset(data=_test_dataset)
xs_test_mnar_90 = torch.stack([_test_dataset[i][0] for i in range(len(_test_dataset))])

pred_errors = []

if model_index == 0 or model_index == 3:
    ys = xs[:, -1]
    ys_test_complete = xs_test_complete[:, -1]

    xs = xs[:, :-1]
    xs_test_complete = xs_test_complete[:, :-1]
    xs_test_mcar_10 = xs_test_mcar_10[:, :-1]
    xs_test_mcar_20 = xs_test_mcar_20[:, :-1]
    xs_test_mcar_30 = xs_test_mcar_30[:, :-1]
    xs_test_mcar_40 = xs_test_mcar_40[:, :-1]
    xs_test_mcar_50 = xs_test_mcar_50[:, :-1]
    xs_test_mcar_60 = xs_test_mcar_60[:, :-1]
    xs_test_mcar_70 = xs_test_mcar_70[:, :-1]
    xs_test_mcar_80 = xs_test_mcar_80[:, :-1]
    xs_test_mcar_90 = xs_test_mcar_90[:, :-1]
    
    xs_test_mar_10 = xs_test_mar_10[:, :-1]
    xs_test_mar_20 = xs_test_mar_20[:, :-1]
    xs_test_mar_30 = xs_test_mar_30[:, :-1]
    xs_test_mar_40 = xs_test_mar_40[:, :-1]
    xs_test_mar_50 = xs_test_mar_50[:, :-1]
    xs_test_mar_60 = xs_test_mar_60[:, :-1]
    xs_test_mar_70 = xs_test_mar_70[:, :-1]
    xs_test_mar_80 = xs_test_mar_80[:, :-1]
    xs_test_mar_90 = xs_test_mar_90[:, :-1]
    
    xs_test_mnar_10 = xs_test_mnar_10[:, :-1]
    xs_test_mnar_20 = xs_test_mnar_20[:, :-1]
    xs_test_mnar_30 = xs_test_mnar_30[:, :-1]
    xs_test_mnar_40 = xs_test_mnar_40[:, :-1]
    xs_test_mnar_50 = xs_test_mnar_50[:, :-1]
    xs_test_mnar_60 = xs_test_mnar_60[:, :-1]
    xs_test_mnar_70 = xs_test_mnar_70[:, :-1]
    xs_test_mnar_80 = xs_test_mnar_80[:, :-1]
    xs_test_mnar_90 = xs_test_mnar_90[:, :-1]
    
    # train a baseline random forest model
    # regressor = LinearRegression()
    regressor = MLPRegressor(
        hidden_layer_sizes=(64,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=321
    )
    regressor.fit(xs_test_complete, ys_test_complete)    
    # Predict and impute the missing coordinates
    ys_predicted = regressor.predict(xs)
    pred_errors.append(((torch.as_tensor(ys) - torch.as_tensor(ys_predicted)) ** 2).mean().item())
    
if model_index == 1:
    ys = xs[:, -1]
    ys_test_complete = xs_test_complete[:, -1]

    xs = xs[:, :-2]
    xs_test_complete = xs_test_complete[:, :-2]
    xs_test_mcar_10 = xs_test_mcar_10[:, :-2]
    xs_test_mcar_20 = xs_test_mcar_20[:, :-2]
    xs_test_mcar_30 = xs_test_mcar_30[:, :-2]
    xs_test_mcar_40 = xs_test_mcar_40[:, :-2]
    xs_test_mcar_50 = xs_test_mcar_50[:, :-2]
    xs_test_mcar_60 = xs_test_mcar_60[:, :-2]
    xs_test_mcar_70 = xs_test_mcar_70[:, :-2]
    xs_test_mcar_80 = xs_test_mcar_80[:, :-2]
    xs_test_mcar_90 = xs_test_mcar_90[:, :-2]
    
    xs_test_mar_10 = xs_test_mar_10[:, :-2]
    xs_test_mar_20 = xs_test_mar_20[:, :-2]
    xs_test_mar_30 = xs_test_mar_30[:, :-2]
    xs_test_mar_40 = xs_test_mar_40[:, :-2]
    xs_test_mar_50 = xs_test_mar_50[:, :-2]
    xs_test_mar_60 = xs_test_mar_60[:, :-2]
    xs_test_mar_70 = xs_test_mar_70[:, :-2]
    xs_test_mar_80 = xs_test_mar_80[:, :-2]
    xs_test_mar_90 = xs_test_mar_90[:, :-2]
    
    xs_test_mnar_10 = xs_test_mnar_10[:, :-2]
    xs_test_mnar_20 = xs_test_mnar_20[:, :-2]
    xs_test_mnar_30 = xs_test_mnar_30[:, :-2]
    xs_test_mnar_40 = xs_test_mnar_40[:, :-2]
    xs_test_mnar_50 = xs_test_mnar_50[:, :-2]
    xs_test_mnar_60 = xs_test_mnar_60[:, :-2]
    xs_test_mnar_70 = xs_test_mnar_70[:, :-2]
    xs_test_mnar_80 = xs_test_mnar_80[:, :-2]
    xs_test_mnar_90 = xs_test_mnar_90[:, :-2]
    
    # train a baseline random forest model
    # regressor = LinearRegression()
    regressor = MLPRegressor(
        hidden_layer_sizes=(64,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=321
    )
    regressor.fit(xs_test_complete, ys_test_complete)    
    # Predict and impute the missing coordinates
    ys_predicted = regressor.predict(xs)
    pred_errors.append(((torch.as_tensor(ys) - torch.as_tensor(ys_predicted)) ** 2).mean().item())


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

elbos = []
_dataset = ToyDataset(data=xs_test_complete)
test_dataloader=torch.utils.data.DataLoader(_dataset,batch_size=1024,shuffle=True,drop_last=True)
elbos.append(model.test_step(test_dataloader))

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

# -----------------------------
# Cluster summary
# -----------------------------
print("Cluster assignments:")
for k, count in zip(*cs.unique(return_counts=True)):
    print(f"Cluster {k.item():>2d}: {count.item():>5d}  ({100*count/len(cs):.1f}%)")

# method 03: latent diffusion model with one chart
# noise schedule for continuous and discrete cases
T = 100
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
    def __init__(self, dim, n_classes=2, hidden=256, n_heads=4):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        self.class_emb = nn.Embedding(n_classes, hidden)

        self.x_proj = nn.Linear(dim, hidden)

        encoder_layer = nn.TransformerEncoderLayer( # original value + attention + multilayer perception
            d_model=hidden, 
            nhead=n_heads, # we use 4 heads, so H / 4 in each attention
            dim_feedforward=4 * hidden, # H -> 4H -> MLP -> 4H -> H
            activation="gelu", # the activation used in MLP
            batch_first=True # specify the order of the input dimensions
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2) # the process is repeated twice

        self.x_head = nn.Linear(hidden, dim)
        self.c_head = nn.Linear(hidden, n_classes)

    def forward(self, x, c, t):
        t = t.float().unsqueeze(-1) / T
        t_emb = self.time_mlp(t)
        c_emb = self.class_emb(c)
        x_emb = self.x_proj(x)

        tokens = torch.stack([x_emb, t_emb, c_emb], dim=1)  # [B, 3, H], 3 is the number of tokens
        h = self.transformer(tokens)

        h = h[:, 0]  # use x token as output anchor

        return self.x_head(h), self.c_head(h)


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
        #print(t)
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
    

# -----------------------------
# Likelihood weights
# -----------------------------
def compute_weights(x_samples, x_seen, mask, sigma):
    diff = (x_samples - x_seen) * mask
    dist2 = (diff ** 2).sum(dim=1)
    #w = torch.exp(-dist2 / (2 * sigma**2))
    log_w = -dist2 / (2 * sigma**2)
    log_w = log_w - log_w.max()
    w = torch.exp(log_w)
    return w

real = xs_test_complete.detach().cpu()

xs_tests = [
            # xs_test_mcar_10, xs_test_mcar_20, xs_test_mcar_30, xs_test_mcar_40, xs_test_mcar_50, 
            # xs_test_mcar_60, xs_test_mcar_70, xs_test_mcar_80, xs_test_mcar_90, 
            # xs_test_mar_10, xs_test_mar_20, xs_test_mar_30, xs_test_mar_40, xs_test_mar_50, 
            # xs_test_mar_60, xs_test_mar_70, xs_test_mar_80, xs_test_mar_90, 
            # xs_test_mnar_10, xs_test_mnar_20, xs_test_mnar_30, xs_test_mnar_40, xs_test_mnar_50,  
            # xs_test_mnar_60, xs_test_mnar_70, xs_test_mnar_80, 
            xs_test_mnar_90, 
            ]
prints = [
            # "MCAR10", "MCAR20", "MCAR30", "MCAR40", "MCAR50", "MCAR60", "MCAR70", "MCAR80", "MCAR90",
            # "MAR10", "MAR20", "MAR30", "MAR40", "MAR50", "MAR60", "MAR70", "MAR80", "MAR90",
            # "MNAR10", "MNAR20", "MNAR30", "MNAR40", "MNAR50", "MNAR60", "MNAR70", "MNAR80", 
            "MNAR90",
        ]

for index in range(len(xs_tests)):
    # print(prints[index])

    xs_test = xs_tests[index]
    N, D = xs_test.shape

    # -----------------------------
    # Create missing mask
    # -----------------------------
    mask = torch.ones_like(xs_test)

    for i in range(N):
        idx = torch.where(torch.isnan(xs_test[i,]))[0]
        mask[i, idx] = 0.0

    x_seen = torch.nan_to_num(xs_test, nan=0.0)
    
    # -----------------------------
    # Sample from joint prior p(z, c)
    # -----------------------------
    M = 10000
    z_new, c_new = sample_04(model_dfzc, M, dim_ld, num_generators)
    # -----------------------------
    # Decode prior samples
    # -----------------------------
    x_new = torch.zeros((M, D), device=z_new.device)

    for k in range(K):
        mk = (c_new == k)
        if mk.sum() == 0:
            continue
        x_new[mk] = model.decoders[k](z_new[mk])


    # -----------------------------
    # 6. SIR imputation per sample
    # -----------------------------
    x_imputed_list = []

    for i in range(N):

        x_seen_i = x_seen[i].unsqueeze(0).repeat(M, 1)
        mask_i = mask[i].unsqueeze(0).repeat(M, 1)

        ws = compute_weights(x_new, x_seen_i, mask_i, sig_hd)
        ws = ws / (ws.sum() + 1e-12)

        idx = torch.multinomial(ws, 1).squeeze(0)
        # idx = torch.argmax(ws)

        z_star = z_new[idx].unsqueeze(0)
        c_star = c_new[idx].item()

        x_full = model.decoders[c_star](z_star)

        # final observation noise (optional but consistent)
        x_full = x_full + sig_hd * torch.randn_like(x_full)

        # fill missing entries
        x_imputed = x_seen[i].clone()
        x_imputed[mask[i] == 0] = x_full[0][mask[i] == 0]

        x_imputed_list.append(x_imputed)


    # -----------------------------
    # 7. Final tensor
    # -----------------------------
    x_imputed = torch.stack(x_imputed_list)
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

    #print(f"RMSE over missing components: {rmse_total.item():.4f}")
    
    X = real.numpy()
    Y = fake.numpy()
    _n = X.shape[0]
    _m = Y.shape[0]
    a = np.ones(_n) / _n
    b = np.ones(_m) / _m
    M = ot.dist(X, Y, metric='euclidean')
    W = ot.emd2(a, b, M)
    #print(f"Wasserstein:{np.sqrt(W)}")
    
    print(f"{rmse_total.item():.4f},{np.sqrt(W):.4f}")
    
    if model_index in [0]:
        # train a baseline random forest model
        regressor = LinearRegression()
        regressor.fit(fake, ys_test_complete)    
        # Predict and impute the missing coordinates
        ys_predicted = regressor.predict(xs)
        pred_errors.append(((torch.as_tensor(ys) - torch.as_tensor(ys_predicted)) ** 2).mean().item())
    
    # _dataset = ToyDataset(data=fake)
    # test_dataloader=torch.utils.data.DataLoader(_dataset,batch_size=1024,shuffle=True,drop_last=True)
    # elbos.append(model.test_step(test_dataloader))
        
print("downstream performance evaluation")
for error in pred_errors:
    print(error)
