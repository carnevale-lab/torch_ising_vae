import numpy as np
from copy import deepcopy

def get_latent_samples(g, mu, sigma, m):
        latent_samples = list()
        for x in range(0, g):
            latent_samples.append([])
        for x in range(0, m):
            samples = np.random.normal(mu, sigma, g)
            for latent_row, sample in zip(latent_samples, samples):
                latent_row.append(sample)
        latent_samples = [[round(float(i), 4) for i in nested] for nested in latent_samples]
        latent_samples = np.array(latent_samples)
        return latent_samples

def prep_bernoulli(z):
    brnll = np.clip(z, 1e-7, 1 - 1e-7)
    return brnll

def generate_seqs(decoder_output, batch_size, latent_dim):
    print("")
    # z = norm.rvs(0., 1., size=(batch_size, latent_dim))
    brnll = prep_bernoulli(decoder_output)
    # c = np.cumsum(brnll, axis=2)
    # c = c/c[:,:,-1,None] # correct for fp error
    r = np.random.rand(50000, 900)
    # seqs = np.sum(r[:,:] > brnll, axis=1, dtype='u1')
    seqs=[]
    for i in range(brnll.shape[0]):
        seqlist = []
        for j in range(brnll.shape[1]):
            if(r[i][j] > brnll[i][j]):
                seqlist.append(0)
            elif(r[i][j] < brnll[i][j]):
                seqlist.append(1)
        seqs.append(deepcopy(seqlist))
    return np.array(seqs)





