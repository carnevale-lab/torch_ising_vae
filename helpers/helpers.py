import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly
import plotly.graph_objects as go

def save_pickle(model, out_name):
    path = "pickles/" +out_name+ ".pth"
    torch.save(model.state_dict(), path)

def load_pickle(path, model):
    temp = "pickles/"
    temp += path
    model.load_state_dict(torch.load(temp))

def save_npy(path, arr):
    np.save(path, arr)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
