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
from helpers.helpers import check_act

class Plotter():
    def __init__(self, conf):
        self.epochs = conf.EPOCHS
        self.batch_size = conf.BATCH_SIZE
        self.hidden_dim = conf.LAYERS
        self.x_dim  = conf.IN_DIM
        self.n_count = conf.NUM_NODES
        self.l_dim = conf.LATENT
        self.activation = check_act(conf.ACTIVATE)

    def loss_plot(self, train_loss_arr, val_loss_arr, out_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(self.epochs),train_loss_arr, label="Training Loss")    
        ax.plot(range(self.epochs),val_loss_arr, label="Validation Loss")
        plt.legend(loc='upper right')
        plt.savefig(out_name + "/loss_plot.png")


    def init_mag(arr):
        mag = []
        for i in range(len(arr)): # Total magnetization for t-SNE coloring
            up = np.sum(arr.numpy()[i])
            down = -(900 - up)
            mag.append(-(up+down)/900)

        return mag

    def latent_plot(mean, log_var, mag, out_name):
        Path(out_name+"/latent_plots").mkdir(parents=True, exist_ok=True)
        mean = mean.cpu().numpy()
        log_var = log_var.cpu().numpy()
        check = []
        for i in range(self.l_dim):
            for j in range(self.l_dim):
                if i == j or ((i,j) or (j,i)) in check:
                    continue
                else:
                    x = list(mean[:,i])
                    y = list(mean[:,j])
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    scat = ax.scatter(x,y,c=mag)
                    plt.colorbar(scat)
                    plt.savefig(out_name+"/latent_plots/latent_"+str(i)+"+"+str(j)+".png")
                    check.append((i,j))
                    plt.close('all')

    def tsne_plot(mean,mag,out_name):
        mean = mean.cpu().numpy()
        np.random.shuffle(mean)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        tsne = TSNE(n_components=3,learning_rate=100)
        proj = tsne.fit_transform(mean[:15000])
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
        plotly.offline.plot(fig, filename=out_name+'/latent_plots/latent_TSNE.html')

    def hamming_plot(out_name):
        seqs = loadSeqs(out_name+"/orig.txt", names="01")[0][0:]
        h = histsim(seqs).astype(float)
        h = h/np.sum(h)
        rev_h = h[::-1]

        seqs1 = loadSeqs(out_name+"/pred.txt", names="01")[0][0:]
        h1 = histsim(seqs1).astype(float)
        h1 = h1/np.sum(h1)
        rev_h1 = h1[::-1]

        fig, ax = plt.subplots()
        ax.plot(rev_h, label="Original")
        ax.plot(rev_h1, label="Predicted")
        plt.legend(loc="best")
        plt.savefig(out_name+"/hamming_plot.png")

    def plot_covars():
        print("\t\t\t\tplotting covars:\t")
        fig, ax = plt.subplots(figsize=(3,3))
        marker_size = 1
        start = -0.35
        end = 0.40
        x_tick_range = np.arange(start, end, 0.1)
        y_tick_range = np.arange(start, end, 0.1)
        box_props = dict(boxstyle="round", facecolor="wheat")
        target_covars = np.load("E20_B500_D3_N50_L7_ELU/covars_Original.npy")
        target_masked = np.ma.masked_inside(target_covars, -0.01, 0.01).ravel()
        target_covars = target_covars.ravel()
        pred_covars = np.load("E20_B500_D3_N50_L7_ELU/covars_Predicted.npy")
        pred_masked = np.ma.masked_inside(pred_covars, -0.01, 0.01).ravel()
        pred_covars = pred_covars.ravel()

        other_covars = {"Original" : target_covars, "Predicted" : pred_covars}
        z_order= {"Original" : -1 , "Predicted" : -25}
        colors = {"Original" : "black", "Predicted" : "cyan"}
        # for label, covars in other_covars.items():
        label = "Predicted"
        print("\t\t\t\t\tcovar corrs for:\ttargetSeqs", "\t\t", label)
        #pearson_r, pearson_p = pearsonr(covars_a, covars_b)
        #pearson_r = pearsonr(target_covars, covars)
        pearson_r, pearson_p = pearsonr(pred_covars, pred_covars)
        #print(pearson_r) 
        c = colors["Predicted"]
        print(pearson_r, pearson_p)
        label_text = label
        label_text = label + ", " + r"$\rho$ = " + str(round(pearson_r, 2))     # orig with rho
        #plt.plot(target_masked, covars, 'o', markersize=marker_size, color=c, alpha=self.alpha, label=label_text)
        ax.plot(pred_covars, pred_covars, 'o', markersize=marker_size, color=c, label=label_text, zorder=-1, alpha=0.9)
        #plt.plot(target_masked, covars, 'o', markersize=marker_size, color=c)

        ax.set_rasterization_zorder(0)
        title_text = "Covariance Correlations Scatterplot\n"
        box_y = 0.40
        box_x = -0.35
        xlabel= "Original"
        # if "nat" in self.synth_nat:
        #     xlabel = "Nat-Target Covariances"
        # else:
        #     xlabel = "Synth-Target Covariances"
        pylab.xlabel(xlabel, fontsize=11)
        pylab.ylabel("GPSM Covariances", fontsize=11)
        #pylab.title(self.which_size, fontsize=self.title_size)
        pylab.xticks(x_tick_range, rotation=45, fontsize=9)
        pylab.yticks(y_tick_range, fontsize=9)
        pylab.tick_params(direction='in', axis='both', which='major', labelsize=9, length=4, width=0.6)
        lim_start = -0.35
        lim_end = 0.40
        pylab.xlim((lim_start, lim_end))
        pylab.ylim((lim_start, lim_end))
        pylab.tight_layout()
        file_name = "covars.png"
        leg_fontsize = 6
        pylab.legend(fontsize=leg_fontsize, loc="upper left", title_fontsize=leg_fontsize, frameon=False)
        save_name = "E20_B500_D3_N50_L7_ELU/" + file_name
        pylab.savefig(save_name, dpi=500, format='png')
        pylab.close()
        print("\t\tcompleted: plotting covars")
