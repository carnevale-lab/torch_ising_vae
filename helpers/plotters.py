import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from pathlib import Path
from helpers.mi3gpu.utils.seqtools import histsim
from helpers.mi3gpu.utils.seqload import loadSeqs, writeSeqs


def loss_plot(epochs, train_loss_arr, val_loss_arr, out_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(epochs),train_loss_arr, label="Training Loss")    
    ax.plot(range(epochs),val_loss_arr, label="Validation Loss")
    plt.legend(loc='upper right')
    plt.savefig(f"training_plots/" + out_name + ".png")


def init_mag(arr):
    mag = []
    for i in range(len(arr)): # Total magnetization for t-SNE coloring
        up = np.sum(arr.numpy()[i])
        down = -(900 - up)
        mag.append(-(up+down)/900)

    return mag

def latent_plot(mean, log_var, latent_dim, mag, out_name):
    Path("latent_plots/"+out_name).mkdir(parents=True, exist_ok=True)
    mean = mean.cpu().numpy()
    log_var = log_var.cpu().numpy()
    check = []
    for i in range(latent_dim):
        for j in range(latent_dim):
            if i == j or ((i,j) or (j,i)) in check:
                continue
            else:
                x = list(mean[:,i])
                y = list(mean[:,j])
                fig = plt.figure()
                ax = fig.add_subplot(111)
                scat = ax.scatter(x,y,c=mag)
                plt.colorbar(scat)
                plt.savefig("latent_plots/"+out_name+"/latent_"+str(i)+"+"+str(j)+".png")
                check.append((i,j))
                plt.close('all')

def tsne_plot(mean,mag,out_name):
    mean = mean.cpu().numpy()
    np.random.shuffle(mean)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    tsne = TSNE(n_components=3,learning_rate=100)
    proj = tsne.fit_transform(mean[:1000])
    x = list(proj[:,0])
    y = list(proj[:,1])
    z = list(proj[:,2])
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color=mag,
            colorscale='spectral',
            opacity=0.6
        )
    )])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(showlegend=True)
    fig.show()
    plotly.offline.plot(fig, filename='latent_plots/'+out_name+'/latent_' +out_name+ '.html')

def hamming_plot(out_name):
    seqs = loadSeqs("helpers/hamstxt/orig.txt", names="01")[0][0:]
    h = histsim(seqs).astype(float)
    h = h/np.sum(h)
    rev_h = h[::-1]

    seqs1 = loadSeqs("helpers/hamstxt/pred.txt", names="01")[0][0:]
    h1 = histsim(seqs1).astype(float)
    h1 = h1/np.sum(h1)
    rev_h1 = h1[::-1]

    fig, ax = plt.subplots()
    ax.plot(rev_h, label="Original")
    ax.plot(rev_h1, label="Predicted")
    plt.legend(loc="best")
    plt.savefig("hamming/"+out_name+"_hamming.png")
