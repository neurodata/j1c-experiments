from multiprocessing import Pool
from os import path
import _pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from graspy.utils import pass_to_ranks as ptr
from graspy.embed import AdjacencySpectralEmbed, select_dimension
from graspy.plot import heatmap, pairplot
from mgcpy.independence_tests.mgc import MGC
from mgcpy.independence_tests.dcorr import DCorr


def k_sample_transform(x, y):
    u = np.concatenate([x, y], axis=0)
    v = np.concatenate([np.repeat(1, x.shape[0]), np.repeat(2, y.shape[0])], axis=0)
    if len(u.shape) == 1:
        u = u[..., np.newaxis]
    if len(v.shape) == 1:
        v = v[..., np.newaxis]
    return u, v

def get(A1, A2, n=50, binarize=False, run_ptr=False):
    #A1 = .5*(A1 + np.transpose(A1))
    #A2 = .5*(A2 + np.transpose(A2))
    if binarize:
        print('binarized')
        A1[A1>0] = 1
        A2[A2>0] = 1
    elif run_ptr:
        print('ptred')
        A1 = ptr(A1)
        A2 = ptr(A2)
    else:
        print('neither')
    num_dims1 = select_dimension(A1.astype(float))[0][-1]
    num_dims2 = select_dimension(A2.astype(float))[0][-1]
    n_components = max(num_dims1, num_dims2)
    print(n_components)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    A1 = ase.fit_transform(A1)
    A2 = ase.fit_transform(A2)
    A1c = np.concatenate((A1[0], A1[1]), axis=1)
    A2c = np.concatenate((A2[0], A2[1]), axis=1)
    print(A1c.shape, A2c.shape)
    return A1c, A2c

def run_tests(X, Y, fname='tests', use_cached=True):
    if use_cached:
        bvals = pkl.load(open(f'../data/bd{fname}.pkl','rb'))
        uvals = pkl.load(open(f'../data/ud{fname}.pkl','rb'))
        mvals = pkl.load(open(f'../data/m{fname}.pkl','rb'))

    if path.exists(f'../data/bd{fname}.pkl') and use_cached:
        print('loading biased DCorr')
        bvals = pkl.load(open(f'../data/bd{fname}.pkl','rb'))
    else:
        print('running biased DCorr')
        biased_dcorr = DCorr(which_test='biased')
        bdt, bdm = biased_dcorr.test_statistic(X, Y)
        bdp, bdmp = biased_dcorr.p_value(X, Y, is_fast=False)
        bvals = {'statistic': [bdt, bdm],
                'p_value': [bdp, bdmp]}
        pkl.dump(bvals, open(f'../data/bd{fname}.pkl','wb'))

    if path.exists(f'../data/ud{fname}.pkl') and use_cached:
        print('loading unbiased DCorr')
        uvals = pkl.load(open(f'../data/ud{fname}.pkl','rb'))
    else:
        print('running unbiased DCorr')
        unbiased_dcorr = DCorr(which_test='unbiased')
        udt, udm = unbiased_dcorr.test_statistic(X, Y)
        udp, udmp = unbiased_dcorr.p_value(X, Y, is_fast=False)
        uvals = {'statistic': [udt, udm],
                'p_value': [udp, udmp]}
        pkl.dump(uvals, open(f'../data/ud{fname}.pkl','wb'))

    if path.exists(f'../data/m{fname}.pkl') and use_cached:
        print('loading unbiased mgc')
        mvals = pkl.load(open(f'../data/m{fname}.pkl','rb'))
    else:
        print('running unbiased mgc')
        mgc = MGC()
        mt, mm = mgc.test_statistic(X, Y, is_fast=False)
        mp, mmp = mgc.p_value(X, Y, is_fast=False)
        mvals = {'statistic': [mt, mm],
                'p_value': [mp, mmp]}
        pkl.dump(mvals, open(f'../data/m{fname}.pkl','wb'))
    return [bvals, uvals, mvals]

def main(tup, heatmaps=False, pairplots=False, histograms=True):
    fname = tup[0]
    do_bin = tup[1]
    do_ptr = tup[2]
    # load data
    with open('../data/left_adjacency.csv') as csv_file:
        left_adj = np.loadtxt(csv_file, dtype=int)
    with open('../data/right_adjacency.csv') as csv_file:
        right_adj = np.loadtxt(csv_file, dtype=int)
    A, B = get(left_adj, right_adj, binarize=do_bin, run_ptr=do_ptr)
    X, Y = k_sample_transform(A, B)

    if heatmaps:
        # view raw adjacency matrices
        heatmap(left_adj, cbar=False)
        plt.savefig('../figures/left.png',
                     bbox_inches='tight', 
                     transparent=True,
                     pad_inches=0)
        heatmap(right_adj)
        plt.savefig('../figures/right.png',
                     bbox_inches='tight', 
                     transparent=True,
                     pad_inches=0)

    if pairplots:
        # view pairplots of no-ptr and ptr embeddings
        pairplot(X, Y, title=f'Left(red) vs Right(blue) ({fname})')
        plt.savefig(f'../figures/{fname}pairplot.png')

    if histograms:
        val_dict = run_tests(X, Y, fname, use_cached=False)
        bd_stat = val_dict[0]['statistic'][0]
        bd_p = val_dict[0]['p_value'][0]
        ubd_stat = val_dict[1]['statistic'][0]
        ubd_p = val_dict[1]['p_value'][0]
        m_stat = val_dict[2]['statistic'][0]
        m_p = val_dict[2]['p_value'][0]
        
        bdt, bdm = val_dict[0]['statistic']
        udt, udm = val_dict[1]['statistic']
        mt, mm = val_dict[2]['statistic']
        bdp, bdmp = val_dict[0]['p_value']
        udp, udmp = val_dict[1]['p_value']
        mp, mmp = val_dict[2]['p_value']

        # plot test results
        plt.hist(list(bdmp)[0], color='b', alpha=0.5, bins=30, label='biased dcorr', normed=1)
        plt.axvline(bdt, ymax=0.65, color='blue', alpha=0.5)
        plt.hist(list(udmp)[0], color='r', alpha=0.5, bins=30, label='unbiased dcorr', normed=1)
        plt.axvline(udt, ymax=0.65, color='red', alpha=0.5)
        plt.hist(mmp['null_distribution'], color='g', alpha=0.5, bins=30, label='unbiased mgc', normed=1)
        plt.axvline(mt, ymax=0.65, color='green', alpha=0.5)
        plt.xlabel('test statistic')
        plt.ylabel('counts')

        #plt.legend(loc=9)
        plt.ylim([0,250])
        plt.text(-0.0051, 230,
                'Biased DCorr p={:.3f}'.format(bdp),
                fontdict=None, 
                withdash=False,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=9)
        plt.text(.0064, 230, 
                'Unbiased DCorr p={:.3f}'.format(udp),
                fontdict=None,
                withdash=False,
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=9)
        plt.text(.0191, 230, 
                'Unbiased MGC p={:.3f}'.format(mp), 
                fontdict=None, 
                withdash=False, 
                bbox=dict(facecolor='green', alpha=0.5),
                fontsize=9)

        plt.savefig(f'../figures/{fname}nulls.png')

if __name__ == '__main__':
    with Pool() as pool:
        pool.map(main, [('concraw',False, False), ('concbinary', True, False), ('concptr', False, True)])
