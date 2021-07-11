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

def arr_to_str(in_arr):
    strs = []
    for i in range(len(in_arr)):
        strout = ""
        for j in range(len(in_arr[0])):
            strout+= str(in_arr[i][j])
        strs.append(strout)

    return np.array(strs)

def seqs_to_txt(out_name):
    original = np.load("data/ising.npy")
    original = original.astype(int)
    predicted = np.load("generated/"+out_name+"_seqs.npy")
    predicted = predicted.astype(int)

    orig = arr_to_str(original)
    pred = arr_to_str(predicted)
    
    with open("helpers/hamstxt/orig.txt", "w") as f:
        for line in orig:
            f.write(line + '\n')
    f.close()

    with open("helpers/hamstxt/pred.txt", "w") as g:
        for line in pred:
            g.write(line + '\n')
    g.close()

