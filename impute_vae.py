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
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

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
    regressor = LinearRegression()
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
    regressor = LinearRegression()
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
    #print(prints[index])

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

    # Calculate the mean across the batch dimension (dim=0) from the training data
    train_mean = torch.mean(xs, dim=0)
        
    # -----------------------------
    # VAE Iterative Imputation Algorithm
    # -----------------------------

    # Step 1: Initialise missing values
    train_mean = torch.mean(xs, dim=0).to(device)
    train_mean_broadcasted = train_mean.expand(N, -1)
    x_imputed = torch.where(mask == 1.0, xs_test, train_mean_broadcasted)

    # Ensure all structural elements are on the correct device
    x_imputed = x_imputed.to(device)
    xs_test = xs_test.to(device)
    mask = mask.to(device)

    # Hyperparameters for the iterative reconstruction loop
    max_iter = 100 # 10000 no improvement
    tolerance = 1e-12
    prev_error = float("inf")

    # print("Starting Mix-VAE Iterative Imputation...")

    for iteration in range(max_iter):
        # print(iteration)
        # Step 2: Input the data to the trained VAE
        with torch.no_grad():
            # --- 2a: Sample from the latent variable distribution to generate Z given X ---
            pcx = model.classify(x_imputed)
            cs_sampled = pcx.argmax(dim=1, keepdim=True).view(-1)
            
            zs_sampled = torch.empty((N, dim_ld), device=device)
            for c_idx in cs_sampled.unique():
                sub_mask = (cs_sampled == c_idx)
                zs_sampled[sub_mask] = model.decoders[c_idx.item()].backward(x_imputed[sub_mask])
                
            # Inject latent distribution variance
            zs_sampled = zs_sampled + sig_ld * torch.randn_like(zs_sampled)
            
            # --- 2b: Sample from the reconstructed data distribution to generate X_hat given Z ---
            x_reconstructed = torch.empty_like(x_imputed)
            for c_idx in cs_sampled.unique():
                sub_mask = (cs_sampled == c_idx)
                x_reconstructed[sub_mask] = model.decoders[c_idx.item()](zs_sampled[sub_mask])
                # if torch.isnan(x_reconstructed[sub_mask]).any():
                    # nan_rows = torch.isnan(x_reconstructed[sub_mask]).any(dim=1)
                    # nan_z = zs_sampled[sub_mask][nan_rows]
                    # print(f"Num NaN rows: {nan_rows.sum()} / {nan_rows.shape[0]}")
                    # print(f"Z values for NaN output rows: {nan_z}")
                    # print(f"Z stats: min={nan_z.min():.4f}, max={nan_z.max():.4f}, mean={nan_z.mean():.4f}")
                
            # Reconstructed data distribution noise layer
            x_reconstructed = x_reconstructed + sig_hd * torch.randn_like(x_reconstructed)

        # Step 3: Replace the missing values with the reconstructed values, leaving observed unchanged
        # (x_imputed is updated here to be fed into Step 2 of the next loop iteration)
        x_imputed = torch.where(mask == 1.0, xs_test, x_reconstructed)
        # print(torch.isnan(x_imputed).sum())
        

        # Step 4: Compute the reconstruction error of the observed values
        # (Compares the raw reconstruction outputs against the pristine testing data matrix)
        observed_diff = (x_reconstructed - xs_test) * mask
        reconstruction_error = torch.mean(observed_diff ** 2).item()
        
        # Step 5: If the reconstruction error is below a specified tolerance... end.
        if abs(prev_error - reconstruction_error) < tolerance:
            # print(f"Mix-VAE Imputation converged at iteration {iteration}. Observed Error: {reconstruction_error:.6f}")
            break
            
        prev_error = reconstruction_error
    # else:
        # print(f"Mix-VAE Imputation reached max iterations ({max_iter}). Final Observed Error: {prev_error:.6f}")
    
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
    
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="numItermax reached before optimality"
    )
    
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
    
    # if model_index in [0]:
        # # train a baseline random forest model
        # regressor = LinearRegression()
        # regressor.fit(fake, ys_test_complete)    
        # # Predict and impute the missing coordinates
        # ys_predicted = regressor.predict(xs)
        # pred_errors.append(((torch.as_tensor(ys) - torch.as_tensor(ys_predicted)) ** 2).mean().item())
    
    # # _dataset = ToyDataset(data=fake)
    # # test_dataloader=torch.utils.data.DataLoader(_dataset,batch_size=1024,shuffle=True,drop_last=True)
    # # elbos.append(model.test_step(test_dataloader))
        
# print("downstream performance evaluation")
# for error in pred_errors:
    # print(error)