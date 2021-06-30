import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms, utils
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datapath = "data/"
batch_size = 100
hidden_dim = 3
x_dim  = 900
n_count = 50
l_dim = 7
lr = 1e-3
epochs = 30

kwargs = {'num_workers': 1, 'pin_memory': True} 

data_array = torch.from_numpy(np.load(datapath + 'ising.npy'))
print(data_array.size())
train, val = random_split(data_array,[35000,15000])
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
activate=nn.ReLU(), node_count=n_count)

dec = Decoder(dec_dim=hidden_dim, output_dim=x_dim, latent_dim=l_dim,
activate=nn.ReLU(), node_count=n_count)

model = Model(Encoder=enc, Decoder=dec).to(device)

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, x in enumerate(train_loader):
        x = x.float()
        x = x.view(batch_size, x_dim)
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    
print("Finish!!")