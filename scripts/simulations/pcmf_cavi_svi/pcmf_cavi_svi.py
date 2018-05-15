"""
This file compares CAV and SV inference of the non-sparse pCMF on simulated data sets.
We generate data sets with 2 cell types with different levels of separability and 
zero-inflation. 

Here we focus on comparing the running time of both inference algorithms. Does SVI achieve
the same likelihood as CAVI? We limit the running time to 2 hours per experiment. 

INPUT
The file receives as input the data set dimensions, N and P. We always reduce to K=10.

OUTPUT
We output the average log likelihood curves per second and iteration, the 2D PCA plots
of the inferred latent space and the silhouette_score of the K-dimensional latent space.
"""

from pCMF.misc import utils
from pCMF.models.pcmf import cavi_new, svi_new, gibbs

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
sns.set_style('whitegrid')

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy.stats import gamma

import operator
import os


def main():
	newpath = './output'
	if not os.path.exists(newpath):
	    os.makedirs(newpath)

	# Experiment parameters
	N = 1000 # number of observations
	P = 20 # observation space dimensionality
	K = 10 # latent space dimensionality
	C = 2 # number of clusters

	# Generate data set
	z_p = 0.3
	eps = 5.
	Y, D, X, R, V, U, clusters = utils.generate_data(N, P, K, C=C, zero_prob=z_p, eps=eps, return_all=True)

	Y_train, Y_test, U_train, U_test, c_train, c_test = train_test_split(Y, U.T, clusters, test_size=0.2, random_state=42)

	#T = 60. * 60. * 2. # max run time
	T = 2.
	S = 1. # ll sampling rate

	# Run PCA
	pca = PCA(n_components=2).fit_transform(Y_train)

	# Run CAVI and get estimates (pCMF)
	alpha = np.ones((2, K))
	alpha[0, :] = 3.
	alpha[1, :] = 0.5
	beta = np.ones((2, P, K))
	pi_D = np.ones((P,)) * 0.5
	print('CAVI:')
	inf = cavi_new.CoordinateAscentVI(Y_train, alpha, beta, pi_D)
	cavi_ll = inf.run_cavi(n_iterations=100000, empirical_bayes=True, return_ll=True, sampling_rate=S, max_time=T)
	cavi_U = inf.a[0] / inf.a[1] # VI estimate is the mean of the variational approximation
	cavi_pca = PCA(n_components=2).fit_transform(cavi_U)
	
	print('\n')

	# Run SVI and get estimates (pCMF)
	alpha = np.ones((2, K))
	alpha[0, :] = 3.
	alpha[1, :] = 0.5
	beta = np.ones((2, P, K))
	pi_D = np.ones((P,)) * 0.5
	mb = 100
	print('SVI:')
	inf = svi_new.StochasticVI(Y_train, alpha, beta, pi_D)
	svi_ll = inf.run_svi(n_iterations=100000, minibatch_size=mb, empirical_bayes=True, return_ll=True, sampling_rate=S, max_time=T)
	svi_U = inf.a[0] / inf.a[1] # VI estimate is the mean of the variational approximation
	svi_pca = PCA(n_components=2).fit_transform(svi_U)

	print('\n')

	# Plot convergence curves
	plt.plot(cavi_ll[1], label='SVI-{}'.format(mb))
	plt.plot(svi_ll[1], label='CAVI')
	plt.ylabel('Average log-likelihood')
	plt.xlabel('Seconds')
	plt.title('Data set with N={} and P={}'.format(N, P))
	plt.legend()
	plt.savefig('./output/convergence_seconds.png')

	plt.plot(cavi_ll[0], label='SVI-{}'.format(mb))
	plt.plot(svi_ll[0], label='CAVI')
	plt.ylabel('Average log-likelihood')
	plt.xlabel('Iterations')
	plt.title('Data set with N={} and P={}'.format(N, P))
	plt.legend()
	plt.savefig('./output/convergence_iterations.png')

	# Plot 2D PCA scatter plots
	U_list = [pca, cavi_pca, svi_pca]
	title_list = ['PCA', 'CAVI', 'SVI']
	n_results = len(U_list)
	assert len(U_list) == len(title_list)
	fig = plt.figure(figsize=(16, 4))
	s = 30
	alpha = 0.7
	labels=None
	for i in range(len(U_list)):
	    ax = plt.subplot(1, n_results, i+1)
	    handlers = []
	    for c in range(C):
	        h = ax.scatter(U_list[i][c_train==c, 0], U_list[i][c_train==c, 1], s=s, alpha=alpha)
	        handlers.append(h)
	    if labels is not None:
	        ax.legend(handlers, labels, scatterpoints=1)
	    plt.title(title_list[i])
	plt.savefig('./output/2D_scatter.png')

	# Print out silhouette scores to file
	true_silh = silhouette_score(U_train, c_train)
	cavi_silh = silhouette_score(cavi_U, c_train)
	svi_silh = silhouette_score(svi_U, c_train)
	pca_silh = silhouette_score(pca, c_train)
	scores = {'PCA': pca_silh, 'CAVI': cavi_silh, 'SVI': svi_silh}
	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	with open('./output/silhouette_scores.txt', 'w') as f:
		print('Silhouette scores (higher is better):', file=f)
		print('\033[1m- {0}: {1:.3}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
		for score_tp in sorted_scores[1:]:
		    print('- {0}: {1:.3}'.format(score_tp[0], score_tp[1]), file=f)
		print('\nSilhouette of true U:', file=f)
		print('%0.3f' % true_silh, file=f)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(-1)