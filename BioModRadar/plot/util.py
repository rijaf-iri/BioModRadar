import numpy as np

def manage_layout(nbplot, n=5):
    nrep = np.arange(1, n * 2, 2)
    nbseq = np.repeat(np.arange(1, n + 1), nrep, axis=0)
    nc = nbseq[nbplot - 1]
    nb = np.arange(1, nbplot + 1)
    sl = np.split(nb, np.arange(nc, len(nb), nc))
    nr = len(sl)
    return nr, nc
