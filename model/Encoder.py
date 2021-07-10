import torch
import torch.nn as nn

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

    def generator(self, arr, device):
        self.eval()
        with torch.no_grad():
            mean,log_var = self.forward(arr.float().to(device))
        return mean, log_var
    
