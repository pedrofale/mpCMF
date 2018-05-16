import argparse
from lib import cavi, svi
import pandas as pd
import numpy as np

import pCMF

parser = argparse.ArgumentParser()
parser.add_argument("--mb", help="minibatch size")
parser.add_argument("--it", help="number of iterations")
parser.add_argument("--ll", help="return log-likelihood [y/]")
parser.add_argument("--counts", required=True, help="file containing mRNA counts")
parser.add_argument("--annotations", required=True, help="file containing cluster annotations")
args = parser.parse_args()
if args.mb is None:
    mb = 1
else:
    mb = int(args.mb)
if args.it is None:
    it = 10
else:
    it = int(args.it)
return_ll = False
if args.ll=='y':
    return_ll = True

brain_tags = pd.read_csv(args.counts)
brain_tags = brain_tags.drop('Unnamed: 0', axis=1)
info = pd.read_csv(args.annotations, sep='\t')
mapping = [list(info.Sample_Name_s).index(cell_name) if cell_name in list(info.Sample_Name_s) else None for cell_name in list(brain_tags.columns)]
cell_type = info.cell_type_s[mapping]
types = cell_type.unique()

brain_10 = brain_tags[np.sum(brain_tags, axis=1) > 10]

Y = brain_10.T.as_matrix()

# Parameters
N = Y.shape[0]
P = Y.shape[1]
K = 2
C = types.size

alpha = np.ones((2, K))
alpha[0, :] = 3.
alpha[1, :] = 0.5
beta = np.ones((2, P, K))
pi = np.ones((P,)) * 0.5
print('Running SVI with minibatch size = {0} for {1} iterations'.format(mb, it))
inf = svi.StochasticVI(Y, alpha, beta, pi)
svi_ll = inf.run_svi(n_iterations=it, minibatch_size=mb, return_ll=return_ll, sampling_rate=1., max_time=10*60.*60.)
svi_U = inf.a[0] / inf.a[1]
svi_V = inf.b[0] / inf.b[1]

cavi = PCMF(Y, n_components=exp['K'], sampling_rate=settings['options']['S'], max_time=settings['options']['T'], verbose=True)
cavi.infer(n_iterations=1000000)

print('\n')
print('Saving U and V to file...')

np.save('output/svi_U.npy', svi.est_U)
np.save('output/svi_V.npy', svi.est_V)

print('Done.')