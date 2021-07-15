import numpy as np
import pandas as pd
from helpers.mi3gpu.utils import seqload
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pylab
import argparse
import sys

#--------------------------COVARS--------------------------------
def getL(size):
    return int(((1+np.sqrt(1+8*size))//2) + 0.5)

def getLq(J):
    return getL(J.shape[0]), int(np.sqrt(J.shape[1]) + 0.5)

def getUnimarg(ff):
    L, q = getLq(ff)
    ff = ff.reshape((L*(L-1)//2, q, q))
    marg = np.array([np.sum(ff[0], axis=1)] +
                    [np.sum(ff[n], axis=0) for n in range(L-1)])
    return marg/(np.sum(marg, axis=1)[:,None]) # correct any fp errors

def indepF(fab):
    L, q = getLq(fab)
    fabx = fab.reshape((fab.shape[0], q, q))
    fa1, fb2 = np.sum(fabx,axis=2), np.sum(fabx,axis=1)
    fafb = np.array([np.outer(fa, fb).flatten() for fa,fb in zip(fa1, fb2)])
    return fafb

def getM(x, diag_fill=0):
    L = getL(len(x))
    M = np.empty((L,L))
    M[np.triu_indices(L,k=1)] = x
    M = M + M.T
    M[np.diag_indices(L)] = diag_fill
    return M

def get_covars(label, bvms_file, path):
    # randomSeqs of VAE are in parent_dir, all others are in data_home

    save_name = path + "/covars_" + label + ".npy"
    bvms_path = path + "/" + bvms_file
    bimarg = np.load(bvms_path)
    C = bimarg - indepF(bimarg)
    np.save(save_name, C)


#---------------------------------BVMS-------------------------

def compute_bvms(seqs, q, weights, nrmlz=True):
    nSeq, L = seqs.shape
    
    #if weights != '0':
    #    weights = np.load(weights)

    if weights == '0':
        weights = None
    
    if q > 16: # the x + q*y operation below may overflow for u1
        seqs = seqs.astype('i4')

    if nrmlz:
        nrmlz = lambda x: x/np.sum(x, axis=-1, keepdims=True)
    else:
        nrmlz = lambda x: x

    def freqs(s, bins):
        return np.bincount(s, minlength=bins, weights=weights)

    #f = nrmlz(np.array([freqs(seqs[:,i], q) for i in range(L)]))
    ff = nrmlz(np.array([freqs(seqs[:,j] + q*seqs[:,i], q*q) \
                         for i in range(L-1) for j in range(i+1, L)]))
    return ff

def get_bvms(label, msa_file, source, dest, A, num_seqs):
    # randomSeqs of VAE are in parent_dir, all others are in data_home
    print("inside get_bvms_phylo, calling get_bvms on ", label)
    #if 'target' in label:
    #    num_seqs = 10000
       
    load_name = source + "/" + msa_file
    bvms_file_name = "bvms_" + label + ".npy"
    save_name = dest + "/" + bvms_file_name
    msa = seqload.loadSeqs(load_name, names="01")[0][:num_seqs]

    print("\t\t\t\timporting msa for:\t", label, "\t", load_name)
    print("\t\t\t\tfinished msa import for:\t", label)
    print("\t\t\t\tcomputing bvms for:\t", label)
    bvms = compute_bvms(msa, A, '0')
    np.save(save_name, bvms)
    print("\t\t\t\tfinished computing bvms for:\t", label)
    return bvms_file_name

#------------------------------------PLOTTING----------------------------------
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
    

# biv_orig = get_bvms("Original", "orig.txt", "E20_B500_D3_N50_L7_ELU", "E20_B500_D3_N50_L7_ELU", 2, 50000)
# get_covars("Original", biv_orig, "E20_B500_D3_N50_L7_ELU")
# biv_pred = get_bvms("Predicted", "pred.txt", "E20_B500_D3_N50_L7_ELU", "E20_B500_D3_N50_L7_ELU", 2, 50000)
# get_covars("Predicted", biv_pred, "E20_B500_D3_N50_L7_ELU")
# plot_covars()