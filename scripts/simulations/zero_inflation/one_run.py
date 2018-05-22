"""
This script proves the importance of considering the model with zero-inflation for zero-inflated data.
We generate data with different levels of zero-inflation and 


Here we focus on comparing the running time of both inference algorithms. Does SVI achieve
the same likelihood as CAVI? We limit the running time to 2 hours per experiment. 

INPUT
The file receives as input the data set dimensions, N and P. We always reduce to K=10.

OUTPUT
We output the average log likelihood curves per second and iteration, the 2D TSNE plots
of the inferred latent space and the silhouette_score of the K-dimensional latent space.
"""

from pCMF.misc import utils
from pCMF.models.pcmf.inferences import cavi_new

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy.stats import gamma

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
sns.set_style('whitegrid')

import time

import operator
import sys, os
import argparse
import json

def main():	
	init = time.time()

	newpath = './onerun_output'
	if not os.path.exists(newpath):
		print('Creating output directory...')
		os.makedirs(newpath)
		print('Done.')

	# Experiment parameters
	N = 100 # number of observations
	P = 1000 # observation space dimensionality
	K = 10 # latent space dimensionality
	C = 3 # number of clusters

	T = 60 * 5
	S = 1

	z_p = 0.3 # none, low, high ZI
	eps = 5. # low, high separability

	Y, D, X, R, V, U, clusters = utils.generate_data(N, P, K, C=C, zero_prob=z_p,
                                                 eps=eps, return_all=True)

	Y_train, Y_test, U_train, U_test, c_train, c_test = train_test_split(Y, U.T, clusters, test_size=0.2, random_state=42)

	# Prior parameters
	alpha = np.ones((2, K))
	alpha[0, :] = 3.
	alpha[1, :] = 0.5
	beta = np.ones((2, P, K))
	pi_D = np.ones((P,)) * 0.5

	# Run PCA
	pca_U = PCA(n_components=K).fit_transform(np.log(Y_train + 1.))
	pca_tsne = TSNE(n_components=2).fit_transform(pca_U)

	print('Simple GaP:')
	infgap = cavi_new.CoordinateAscentVI(Y_train, alpha, beta, empirical_bayes=False)
	infgap.run(n_iterations=4000, calc_ll=True, calc_silh=True, clusters=c_train, sampling_rate=S, max_time=T)
	gap_U = infgap.a[0] / infgap.a[1] # VI estimate is the mean of the variational approximation
	gap_V = infgap.b[0] / infgap.b[1]
	gap_S = infgap.estimate_S(infgap.p_S)
	gap_tsne = TSNE(n_components=2).fit_transform(gap_U)

	print('Simple GaP-EB:')
	infgapeb = cavi_new.CoordinateAscentVI(Y_train, alpha, beta, empirical_bayes=True)
	infgapeb.run(n_iterations=4000, calc_ll=True, calc_silh=True, clusters=c_train, sampling_rate=S, max_time=T)
	gapeb_U = infgapeb.a[0] / infgapeb.a[1] # VI estimate is the mean of the variational approximation
	gapeb_V = infgapeb.b[0] / infgapeb.b[1]
	gapeb_S = infgapeb.estimate_S(infgapeb.p_S)
	gapeb_tsne = TSNE(n_components=2).fit_transform(gapeb_U)

	print('Zero-Inflated GaP:')
	infzigap = cavi_new.CoordinateAscentVI(Y_train, alpha, beta, pi_D=pi_D, empirical_bayes=False)
	infzigap.run(n_iterations=4000, calc_ll=True, calc_silh=True, clusters=c_train, sampling_rate=S, max_time=T)
	zigap_U = infzigap.a[0] / infzigap.a[1] # VI estimate is the mean of the variational approximation
	zigap_V = infzigap.b[0] / infzigap.b[1]
	zigap_S = infzigap.estimate_S(infzigap.p_S)
	zigap_tsne = TSNE(n_components=2).fit_transform(zigap_U)

	print('Zero-Inflated GaP-EB:')
	infzigapeb = cavi_new.CoordinateAscentVI(Y_train, alpha, beta, pi_D, empirical_bayes=True)
	infzigapeb.run(n_iterations=4000, calc_ll=True, calc_silh=True, clusters=c_train, sampling_rate=S, max_time=T)
	zigapeb_U = infzigapeb.a[0] / infzigapeb.a[1] # VI estimate is the mean of the variational approximation
	zigapeb_V = infzigapeb.b[0] / infzigapeb.b[1]
	zigapeb_S = infzigapeb.estimate_S(infzigapeb.p_S)
	zigapeb_tsne = TSNE(n_components=2).fit_transform(zigapeb_U)
	
	print('')
	print('Plotting convergence plots and saving to {0}...'.format(newpath))
	fig = plt.figure(figsize=(12, 4))

	ax = plt.subplot(1, 2, 1)
	ax.plot(infgap.ll_time, label='GaP')
	ax.plot(infgapeb.ll_time, label='GaP-EB')
	ax.plot(infzigap.ll_time, label='ZIGaP')
	ax.plot(infzigapeb.ll_time, label='ZIGaP-EB')
	plt.ylabel('Average log-likelihood')
	plt.xlabel('Seconds(/{0})'.format(S))

	ax = plt.subplot(1, 2, 2)
	ax.plot(infgap.silh_time, label='GaP')
	ax.plot(infgapeb.silh_time, label='GaP-EB')
	ax.plot(infzigap.silh_time, label='ZIGaP')
	ax.plot(infzigapeb.silh_time, label='ZIGaP-EB')
	plt.ylabel('Silhouette of latent space')
	plt.xlabel('Seconds(/{0})'.format(S))

	plt.legend(loc='upper left', bbox_to_anchor=[1., 1.], frameon=True)
	plt.suptitle('Data set with N={} and P={}'.format(N, P), fontsize=14)
	plt.subplots_adjust(top=0.85)

	plt.savefig('{0}/convergence.png'.format(newpath))

	print('Done.')

	print('')
	gap_silh = silhouette_score(gap_U, c_train)
	gapeb_silh = silhouette_score(gapeb_U, c_train)
	zigap_silh = silhouette_score(zigap_U, c_train)
	zigapeb_silh = silhouette_score(zigapeb_U, c_train)
	pca_silh = silhouette_score(pca_U, c_train)

	scores = {'GaP': gap_silh, 'GaP-EB': gapeb_silh, 'ZIGaP': zigap_silh, 'ZIGaP-EB': zigapeb_silh, 'PCA': pca_silh}

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('Silhouette scores (higher is better):')
	print('\033[1m- {0}: {1:.3}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]))
	for score_tp in sorted_scores[1:]:
	    print('- {0}: {1:.3}'.format(score_tp[0], score_tp[1]))

	print('')
	print('Plotting TSNE maps and saving to {0}...'.format(newpath))

	# Plot in decreasing silhouette order
	U_list = [gap_tsne, gapeb_tsne, zigap_tsne, zigapeb_tsne, pca_tsne]
	title_list = ['GaP', 'GaP-EB', 'ZIGaP', 'ZIGaP-EB', 'PCA']

	assert len(U_list) == len(title_list)

	n_results = len(U_list)

	fig = plt.figure(figsize=(16, 4))

	s = 30
	alpha = 0.7
	labels=None
	for i in range(len(U_list)):
	    ax = plt.subplot(1, n_results, i+1)
	    handlers = []
	    for c in range(C):
	        h = ax.scatter(U_list[title_list.index(sorted_scores[i][0])][c_train==c, 0], U_list[title_list.index(sorted_scores[i][0])][c_train==c, 1], s=s, alpha=alpha)
	        handlers.append(h)
	    if labels is not None:
	        ax.legend(handlers, labels, scatterpoints=1)
	    plt.title(sorted_scores[i][0])
	plt.savefig('{0}/tsne.png'.format(newpath))

	print('Done.')

	elapsed = time.time() - init
	print('\nScript finished in {0:.0f} seconds.'.format(elapsed))

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.stderr.write("\nUser interrupt!\n")
		sys.exit(-1)