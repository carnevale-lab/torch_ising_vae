import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torchvision import transforms, utils
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from config import Config
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

print("hi")

datapath = "data/"
batch_size = conf.BATCH_SIZE
hidden_dim = conf.LAYERS
x_dim  = conf.IN_DIM
n_count = conf.NUM_NODES
l_dim = conf.LATENT
lr = 1e-3
epochs = conf.EPOCHS

kwargs = {'num_workers': 1, 'pin_memory': True} 

data_array = torch.from_numpy(np.load(datapath + 'ising.npy'))
print(data_array.size())
train, val = random_split(data_array,[25000,25000])
print(len(train))
print(len(val))

train_loader = DataLoader(train, batch_size=batch_size)
val_loader  = DataLoader(val,  batch_size=batch_size)

class Encoder(nn.Module):
    
    def __init__(self, enc_dim, input_dim, latent_dim, activate, node_count):
        super(Encoder, self).__init__()

        self.enc_dict = nn.ModuleDict({"input": nn.Linear(input_dim, node_count)})
        self.dim = enc_dim
        
        for i in range(enc_dim):
            self.enc_dict.add_module("enc_"+str(i+1), nn.Linear(node_count, node_count))

        self.enc_dict.add_module("mean", nn.Linear(node_count, latent_dim))
        self.enc_dict.add_module("var", nn.Linear(node_count, latent_dim))

        self.activation = activate
        self.training = True
        
    def forward(self, x):
        h_ = self.activation(self.enc_dict["input"](x))

        for i in range(self.dim):
            h_ = self.activation(self.enc_dict["enc_"+str(i+1)](h_))

        mean = self.enc_dict["mean"](h_)
        log_var = self.enc_dict["var"](h_)                    
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, dec_dim, output_dim, latent_dim, activate, node_count):
        super(Decoder, self).__init__()

        self.dec_dict = nn.ModuleDict({"z": nn.Linear(latent_dim, node_count)})
        self.dim = dec_dim

        for i in range(dec_dim):
            self.dec_dict.add_module("dec_"+str(i+1), nn.Linear(node_count, node_count))
                        
        self.dec_dict.add_module("output", nn.Linear(node_count, output_dim))
        
        self.activation = activate
        
    def forward(self, x):
        h = self.activation(self.dec_dict["z"](x))

        for i in range(self.dim):
            h = self.activation(self.dec_dict["dec_"+str(i+1)](h))
        
        x_hat = torch.sigmoid(self.dec_dict["output"](h))
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var

enc = Encoder(enc_dim=hidden_dim, input_dim=x_dim, latent_dim=l_dim,
activate=activation, node_count=n_count)

dec = Decoder(dec_dim=hidden_dim, output_dim=x_dim, latent_dim=l_dim,
activate=activation, node_count=n_count)

model = Model(Encoder=enc, Decoder=dec).to(device)

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


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
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.plot(range(epochs),train_loss_arr, label="Training Loss")    
ax.plot(range(epochs),val_loss_arr, label="Validation Loss")
plt.legend(loc='upper right')
plt.savefig("loss.png")

print("Finish!")
