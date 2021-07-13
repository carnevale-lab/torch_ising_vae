import torch
import torch.nn as nn
import torch.nn.functional as F
from synth_seqs import get_latent_samples, generate_seqs
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Model import Model
from copy import deepcopy

from torch.optim import Adam
from torchvision import transforms, utils
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

from pathlib import Path
from config import Config
from helpers.helpers import save_pickle, load_pickle, loss_function, save_npy, seqs_to_txt, check_act
from helpers.plotters import loss_plot, init_mag, latent_plot, tsne_plot, hamming_plot
import sys

import matplotlib.pyplot as plt
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf_path = "configs/" + str(sys.argv[1])
conf = Config.from_json_file(Path(conf_path))

datapath = "data/"
lr = 1e-3
out_name = "E"+str(conf.EPOCHS)+"_B"+str(conf.BATCH_SIZE)+"_D"+str(conf.LAYERS)+"_N"+str(conf.NUM_NODES)+"_L"+str(conf.LATENT)+"_"+conf.ACTIVATE
Path(out_name).mkdir(parents=True, exist_ok=True)
kwargs = {'num_workers': 1, 'pin_memory': True} 

data_array = torch.from_numpy(np.load(datapath + 'ising.npy'))
train, val = random_split(data_array,[25000,25000])

train_loader = DataLoader(train, batch_size=conf.BATCH_SIZE)
val_loader  = DataLoader(val,  batch_size=conf.BATCH_SIZE)
test_loader = DataLoader(data_array, batch_size=conf.BATCH_SIZE)

enc = Encoder(enc_dim=conf.LAYERS, input_dim=conf.IN_DIM, latent_dim=conf.LATENT,
activate=check_act(conf.ACTIVATE), node_count=conf.NUM_NODES)

dec = Decoder(dec_dim=conf.LAYERS, output_dim=conf.IN_DIM, latent_dim=conf.LATENT,
activate=check_act(conf.ACTIVATE), node_count=conf.NUM_NODES)

model = Model(Encoder=enc, Decoder=dec, device=device, conf=conf).to(device)

if len(sys.argv) == 3:
    load_pickle(model, out_name)
    print(model)
    summary(model,col_names=["kernel_size", "num_params"])

elif len(sys.argv) == 2:
    print(model)
    summary(model,col_names=["kernel_size", "num_params"])
    optimizer = Adam(model.parameters(), lr=lr)
    print("Start training VAE...")
    model.train()

    train_loss_arr, val_loss_arr = model.trainer_func(optimizer, train_loader, val_loader)
    save_pickle(model, out_name)

    loss_plot(conf.EPOCHS, train_loss_arr, val_loss_arr, out_name)

mean, log_var = enc.generator(data_array, device)
print("Enc Generation Complete")
gend = dec.generator(model.l_dim, model.batch_size, device)
save_npy(out_name+"/genSeqs.npy", gend)
print("Dec Generation Complete")

mag = init_mag(data_array)

if conf.LPLOT:
    print("Beginning Latent Plot")
    latent_plot(mean,log_var,model.l_dim, mag, out_name)
    print("Latent Plot Complete")

if conf.TSNE:
    print("Beginning TSNE Plot")
    tsne_plot(mean, mag, out_name)
    print("TSNE Plot Complete")

if conf.HAMMING:
    print("Beginning Hamming Distance Plot")
    seqs_to_txt(out_name)
    hamming_plot(out_name)
    print("Hamming Distance Plot Complete")

print("Finish!")
