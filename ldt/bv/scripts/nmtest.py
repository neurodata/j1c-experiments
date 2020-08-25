import _pickle as pkl
from joblib import Parallel, delayed
import numpy as np
np.random.seed(8888)
import matplotlib.pyplot as plt
import seaborn as sns
from graspy.inference import LatentDistributionTest
from graspy.simulations import sbm
from tqdm import tqdm
sns.set()

def power(A, B, iters=100):
    p = np.zeros(iters)
    for i in range(iters):
        ldt = LatentDistributionTest(n_components=2, which_test='dcorr', graph=True)
        p[i] = ldt.fit(A, B)
    return p

def get_power_for_n(n):
    P = np.array([[0.9, 0.2],[0.2, 0.7]])
    k = 2
    ms = []
    for m in range(n, n+200, 50):
        cn = [n//k] * k
        cm = [m//k] * k
        A = sbm(cn, P)
        B = sbm(cm, P)
        p = power(A, B)
        ms.append(p)
    return ms

plist = Parallel(n_jobs=10)(delayed(get_power_for_n)(i) for i in tqdm(range(100, 500, 100)))
pkl.dump(plist, open('nmplist.pkl','wb'))
