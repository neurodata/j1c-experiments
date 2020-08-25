import _pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from graspy.utils import pass_to_ranks as ptr
from graspy.embed import AdjacencySpectralEmbed, select_dimension
from graspy.plot import heatmap, pairplot
from graspy.inference import LatentDistributionTest
from scipy.stats import gaussian_kde as gkde
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
sns.set()

def k_sample_transform(x, y):
    u = np.concatenate([x,y], axis=0)
    v = np.concatenate([np.repeat(1, x.shape[0]), np.repeat(2, y.shape[0])], axis=0)
    if len(u.shape) == 1:
        u = u[..., np.newaxis]
    if len(v.shape) == 1:
        v = v[..., np.newaxis]
    return u,v

def get(A1, A2, n=50, n_comp=None, symmetrize=False, binarize=False, run_ptr=False):
    if symmetrize:
        A1 = .5*(A1 + np.transpose(A1))
        A2 = .5*(A2 + np.transpose(A2))
    if binarize:
        A1[A1>0] = 1
        A2[A2>0] = 1
    elif run_ptr:
        A1 = ptr(A1)
        A2 = ptr(A2)
    if n_comp is None:
        num_dims1 = select_dimension(A1.astype(float))[0][-1]
        num_dims2 = select_dimension(A2.astype(float))[0][-1]
        n_comp = max(num_dims1, num_dims2)
    ase = AdjacencySpectralEmbed(n_components=n_comp)
    A1 = ase.fit_transform(A1)
    A2 = ase.fit_transform(A2)
    if symmetrize:
        return A1, A2
    else:
        A11 = np.concatenate((A1[0], A1[1]), axis=1)
        A21 = np.concatenate((A2[0], A2[1]), axis=1)
        return A11, A21

with open('../data/left_adjacency.csv') as csv_file:
    left_adj = np.loadtxt(csv_file, dtype=int)
with open('../data/right_adjacency.csv') as csv_file:
    right_adj = np.loadtxt(csv_file, dtype=int)


def get_p_parallel():
    A, B = get(left_adj, right_adj, n_comp=None, symmetrize=False, run_ptr=True)
    X, Y = k_sample_transform(A, B)
    ldt = LatentDistributionTest(n_components=None,which_test='dcorr',graph=False)
    p = ldt.fit(X, Y)
    return p

print('running dcorr')
plist_dcorr = Parallel(n_jobs=10)(delayed(get_p_parallel)() for i in tqdm(range(500)))
pkl.dump(plist_dcorr, open('../data/droslist_dcorr.pkl','wb'))
