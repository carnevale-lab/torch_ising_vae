import torch
import torch.nn as nn
from helpers.helpers import loss_function, check_act
from copy import deepcopy

class Model(nn.Module):
    def __init__(self, Encoder, Decoder, device, conf):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device

        self.epochs = conf.EPOCHS
        self.batch_size = conf.BATCH_SIZE
        self.hidden_dim = conf.LAYERS
        self.x_dim  = conf.IN_DIM
        self.n_count = conf.NUM_NODES
        self.l_dim = conf.LATENT
        self.activation = check_act(conf.ACTIVATE)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var

    def trainer_func(self, optimizer, train_loader, val_loader):
        
        train_loss_arr = []
        val_loss_arr = []

        for epoch in range(self.epochs):
            train_loss = 0
            val_loss = 0

            for batch_idx, x in enumerate(train_loader):
                x = x.float()
                x = x.view(self.batch_size, self.x_dim)
                x = x.to(self.device)

                optimizer.zero_grad()

                x_hat, mean, log_var = self(x)
                loss = loss_function(x, x_hat, mean, log_var)
                
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()

            train_loss_arr.append(train_loss)

            for batch_idx, x in enumerate(val_loader):
                x = x.float()
                x = x.view(self.batch_size, self.x_dim)
                x = x.to(self.device)
                x_hat, mean, log_var = self(x)
                loss = loss_function(x, x_hat, mean, log_var)
                val_loss += loss.item()

            val_loss_arr.append(val_loss)
                
            
            print("\tEpoch", epoch + 1, "complete!", "\tTrain Loss: ", train_loss, "\tVal Loss: ", val_loss)

        return train_loss_arr, val_loss_arr

