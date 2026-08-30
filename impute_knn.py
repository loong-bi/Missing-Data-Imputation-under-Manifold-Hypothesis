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
from sklearn.impute import KNNImputer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import ot
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

    
    # 1. Align devices and map training data to numpy
    xs_train_np = xs.detach().cpu().numpy()
    xs_test_np = xs_test.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    # inductive
    # 2. Create a corrupted test matrix where missing slots are explicit NaNs
    # (This is how sklearn's engines recognize missing values)
    x_corrupted_np = np.where(mask_np == 1.0, xs_test_np, np.nan)

    # 3. Initialize the imputer and FIT it strictly on the clean training data
    # It memorizes the training coordinates to build a fast spatial tree
    imputer = KNNImputer(n_neighbors=100)
    imputer.fit(xs_train_np)

    # 4. Transform the corrupted test data using the trained spatial rules
    x_imputed_np = imputer.transform(x_corrupted_np)
    
    
    # transductive
    # 3. CONCATENATE train + test
    # x_all = np.concatenate([xs_train_np, x_corrupted_np], axis=0)
    # # 4. fit on ALL data
    # imputer = KNNImputer(n_neighbors=100)
    # imputer.fit(x_all)
    # # 5. transform only test part (optional but common)
    # x_imputed_all = imputer.transform(x_all)
    # x_imputed_np = x_imputed_all[len(xs_train_np):]
    
    
    
    # 5. Cast back to PyTorch for your plotting block
    x_imputed = torch.from_numpy(x_imputed_np).float().to(device)
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