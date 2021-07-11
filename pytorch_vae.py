import torch
import torch.nn as nn
import torch.nn.functional as F
from synth_seqs import get_latent_samples, generate_seqs
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Model import Model

from torch.optim import Adam
from torchvision import transforms, utils
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

from pathlib import Path
from config import Config
from helpers.helpers import save_pickle, load_pickle, loss_function, save_npy, seqs_to_txt
from helpers.plotters import loss_plot, init_mag, latent_plot, tsne_plot, hamming_plot
import sys

import matplotlib.pyplot as plt
import numpy as np
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf_path = "configs/" + str(sys.argv[1])
conf = Config.from_json_file(Path(conf_path))


if conf.ACTIVATE == "ReLU":
    activation = nn.ReLU()
elif conf.ACTIVATE == "ELU":
    activation = nn.ELU()
elif conf.ACTIVATE == "SELU":
    activation = nn.SELU()

datapath = "data/"
batch_size = conf.BATCH_SIZE
hidden_dim = conf.LAYERS
x_dim  = conf.IN_DIM
n_count = conf.NUM_NODES
l_dim = conf.LATENT
lr = 1e-3
epochs = conf.EPOCHS
out_name = "E"+str(epochs)+"_B"+str(batch_size)+"_D"+str(hidden_dim)+"_N"+str(n_count)+"_L"+str(l_dim)+"_"+conf.ACTIVATE
kwargs = {'num_workers': 1, 'pin_memory': True} 

data_array = torch.from_numpy(np.load(datapath + 'ising.npy'))
train, val = random_split(data_array,[25000,25000])

train_loader = DataLoader(train, batch_size=batch_size)
val_loader  = DataLoader(val,  batch_size=batch_size)
test_loader = DataLoader(data_array, batch_size=batch_size)

enc = Encoder(enc_dim=hidden_dim, input_dim=x_dim, latent_dim=l_dim,
activate=activation, node_count=n_count)

dec = Decoder(dec_dim=hidden_dim, output_dim=x_dim, latent_dim=l_dim,
activate=activation, node_count=n_count)

model = Model(Encoder=enc, Decoder=dec, device=device).to(device)

if len(sys.argv) > 2:
    load_pickle(sys.argv[2],model)
    print(model)
    summary(model,col_names=["kernel_size", "num_params"])

elif len(sys.argv) == 2:
    print(model)
    summary(model,col_names=["kernel_size", "num_params"])
    optimizer = Adam(model.parameters(), lr=lr)

    print("Start training VAE...")
    model.train()

    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        for batch_idx, x in enumerate(train_loader):
            x = x.float()
            x = x.view(batch_size, x_dim)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss_arr.append(train_loss)

        for batch_idx, x in enumerate(val_loader):
            x = x.float()
            x = x.view(batch_size, x_dim)
            x = x.to(device)
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            val_loss += loss.item()

        val_loss_arr.append(val_loss)
        
        print("\tEpoch", epoch + 1, "complete!", "\tTrain Loss: ", train_loss, "\tVal Loss: ", val_loss)

    save_pickle(model, out_name+"_pickle")

    loss_plot(epochs, train_loss_arr, val_loss_arr, "loss_plot_" + out_name)

mean, log_var = enc.generator(data_array, device)
print("Enc Generation Complete")
mag = init_mag(data_array)
print("Mag Init Complete")
latent_plot(mean,log_var,l_dim, mag, out_name)
print("Latent Plot Complete")
tsne_plot(mean, mag, out_name)
print("TSNE Plot Complete")
gend = dec.generator(l_dim, batch_size, device)
save_npy("generated/"+out_name+"_seqs.npy", gend)
print("Dec Generation Complete")
seqs_to_txt(out_name)
hamming_plot(out_name)
print("Hamming Distance Plot Complete")

print("Finish!")
