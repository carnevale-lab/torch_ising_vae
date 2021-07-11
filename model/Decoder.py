import torch
import torch.nn as nn
from synth_seqs import get_latent_samples, generate_seqs

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

    def generator(self,l_dim, batch_size, device):
        self.eval()
        with torch.no_grad():
            latent_samples = get_latent_samples(50000,0,1,l_dim)
            out = self.forward(torch.from_numpy(latent_samples).float().to(device))
            gend = generate_seqs(out.cpu().numpy(), batch_size, l_dim)
            return gend
    