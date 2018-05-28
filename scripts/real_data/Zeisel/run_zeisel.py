from pCMF.misc import utils, plot_utils
from pCMF.models.pcmf.inferences import cavi_new, svi_new

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import seaborn as sns
sns.set_style('whitegrid')

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
from scipy.stats import gamma

import time
import operator
import sys, os


def main():	
	init = time.time()

	newpath = './output'
	if not os.path.exists(newpath):
		print('Creating output directory...')
		os.makedirs(newpath)
		print('Done.\n')

	data_path = '../../../../data/Zeisel/expression_mRNA_17-Aug-2014.txt'
	tsne_filename = "tsne.png"
	convergence_filename = "convergence.png"

	# Pre-process data
	print('Pre-processing data from {0}...'.format(data_path))
	X = pd.read_csv(data_path, sep="\t", low_memory=False).T
	clusters = np.array(X[7], dtype=str)[2:]
	precise_clusters = np.array(X[0], dtype=str)[2:]
	celltypes, labels = np.unique(clusters, return_inverse=True)
	_, precise_labels = np.unique(precise_clusters, return_inverse=True)
	gene_names = np.array(X.iloc[0], dtype=str)[10:]
	X = X.loc[:, 10:]
	X = X.drop(X.index[0])
	expression = np.array(X, dtype=np.int)[1:]

	# Keep the most variable genes according to the biscuit paper
	selected = np.std(expression, axis=0).argsort()[-558:][::-1]
	expression = expression[:, selected]
	gene_names = gene_names[selected].astype(str)

	# Train/test split
	X_train, X_test, clusters_train, clusters_test, c_train, c_test, cp_train, cp_test = train_test_split(expression, clusters, labels, precise_labels)
	P = X_train.shape[1] # number of genes
	C = np.unique(c_train).size # number of clusters
	print('Done.\n')

	# Experiment parameters
	T = 60. * 60 * 5.
	Tm, Ts = divmod(T, 60)
	Th, Tm = divmod(Tm, 60)
	S = 30.
	max_iter = 1000000
	K = 10 # latent space dimensionality
	param_dict = {'T': '{0:.0f}h{1:.0f}m{2:.0f}s'.format(Th, Tm, Ts), 'S' : S, 'max_iter': max_iter, 'K': K}
	print('Experiment parameters:')
	for param in param_dict:
	    print('{0}: {1}'.format(param, param_dict[param]))
	print('')

	# Run PCA
	print('PCA...')
	pca_U = PCA(n_components=K).fit_transform(np.log(X_train + 1.))
	pca_tsne = TSNE(n_components=2).fit_transform(pca_U)
	print('Done.\n')

	# Prior parameters for GaP and ZIGaP
	alpha = np.abs(np.ones((2, K)) + np.random.rand(2, K))
	beta = np.abs(np.ones((2, P, K)) + np.random.rand(2, P, K))
	logit_pi_D = np.random.rand(P)
	pi_D = np.exp(logit_pi_D) / (1. + np.exp(logit_pi_D))

	# Run GaP
	mb_1 = 100
	print('Stochastic Zero-Inflated GaP-EB w/ minibatch size = {0}...'.format(mb_1))
	infgapeb = svi_new.StochasticVI(X_train, alpha, beta, pi_D=pi_D, minibatch_size=mb_1, empirical_bayes=True)
	infgapeb.run(n_iterations=max_iter, calc_ll=True, calc_silh=True, clusters=c_train, sampling_rate=S, max_time=T)
	gapeb_U = infgapeb.a[0] / infgapeb.a[1] # VI estimate is the mean of the variational approximation
	gapeb_V = infgapeb.b[0] / infgapeb.b[1]
	gapeb_D = infgapeb.estimate_D(infgapeb.p_D)
	gapeb_S = infgapeb.estimate_S(infgapeb.p_S)
	gapeb_tsne = TSNE(n_components=2).fit_transform(gapeb_U)
	print('Done.\n')

	# Run ZIGaP
	mb_2 = 500
	print('Stochastic Zero-Inflated GaP-EB w/ minibatch size = {0}...'.format(mb_2))
	infzigapeb = svi_new.StochasticVI(X_train, alpha, beta, pi_D=pi_D, minibatch_size=mb_2, empirical_bayes=True)
	infzigapeb.run(n_iterations=max_iter, calc_ll=True, calc_silh=True, clusters=c_train, sampling_rate=S, max_time=T)
	zigapeb_U = infzigapeb.a[0] / infzigapeb.a[1] # VI estimate is the mean of the variational approximation
	zigapeb_V = infzigapeb.b[0] / infzigapeb.b[1]
	zigapeb_D = infzigapeb.estimate_D(infzigapeb.p_D)
	zigapeb_S = infzigapeb.estimate_S(infzigapeb.p_S)
	zigapeb_tsne = TSNE(n_components=2).fit_transform(zigapeb_U)
	print('Done.\n')

	# Plot convergence curves (average train data log-likelihood and silhouette scores)
	print('Saving convergence curves to {0}/{1}...'.format(newpath, convergence_filename))
	fig = plt.figure(figsize=(12, 4))
	ax = plt.subplot(1, 2, 1)
	ax.plot(infgapeb.ll_time, label='ZIGaP-EB-{}'.format(mb_1))
	ax.plot(infzigapeb.ll_time, label='ZIGaP-EB-{}'.format(mb_2))
	plt.ylabel('Average log-likelihood')
	plt.xlabel('Seconds(*{0})'.format(S))
	ax = plt.subplot(1, 2, 2)
	ax.plot(infgapeb.silh_time, label='ZIGaP-EB-{}'.format(mb_1))
	ax.plot(infzigapeb.silh_time, label='ZIGaP-EB-{}'.format(mb_2))
	plt.ylabel('Silhouette of latent space')
	plt.xlabel('Seconds(*{0})'.format(S))
	plt.legend(loc='upper left', bbox_to_anchor=[1., 1.], frameon=True)
	plt.suptitle('Zeisel data', fontsize=14)
	plt.subplots_adjust(top=0.85)
	plt.savefig('{0}/{1}'.format(newpath, convergence_filename), bbox_inches='tight')
	print('Done.\n')

	# Compute and print train data log-likelihood
	gapeb_dll = utils.log_likelihood(X_train, gapeb_U, gapeb_V, infgapeb.p_D, gapeb_S)
	zigapeb_dll = utils.log_likelihood(X_train, zigapeb_U, zigapeb_V, infzigapeb.p_D, zigapeb_S)
	scores = {'ZIGaP-EB-{}'.format(mb_1): gapeb_dll, 'ZIGaP-EB-{}'.format(mb_2): zigapeb_dll}
	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
	print('Full data log-likelihood:')
	print('\033[1m- {0}: {1:.3}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]))
	for score_tp in sorted_scores[1:]:
	    print('- {0}: {1:.3}'.format(score_tp[0], score_tp[1]))
	print('')

	# Compute and print test data log-likelihood
	gapeb_holl = infgapeb.predictive_ll(X_test)
	zigapeb_holl = infzigapeb.predictive_ll(X_test)
	scores = {'ZIGaP-EB-{}'.format(mb_1): gapeb_holl, 'ZIGaP-EB-{}'.format(mb_2): zigapeb_holl}
	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
	print('Held-out log-likelihood:')
	print('\033[1m- {0}: {1:.3}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]))
	for score_tp in sorted_scores[1:]:
	    print('- {0}: {1:.3}'.format(score_tp[0], score_tp[1]))
	print('')

	# Compute and print train data silhouette scores
	gapeb_silh = silhouette_score(gapeb_U, c_train)
	zigapeb_silh = silhouette_score(zigapeb_U, c_train)
	pca_silh = silhouette_score(pca_U, c_train)
	scores = {'ZIGaP-EB-{}'.format(mb_1): gapeb_silh, 'ZIGaP-EB-{}'.format(mb_2): zigapeb_silh, 'PCA': pca_silh}
	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
	print('Silhouette scores (higher is better):')
	print('\033[1m- {0}: {1:.3}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]))
	for score_tp in sorted_scores[1:]:
	    print('- {0}: {1:.3}'.format(score_tp[0], score_tp[1]))
	print('')

	# Plot in decreasing silhouette order and save to file
	print('Saving 2D TSNE projection scatter plot to {0}/{1}...'.format(newpath, tsne_filename))
	U_list = [gapeb_tsne, zigapeb_tsne, pca_tsne]
	title_list = ['ZIGaP-EB-{}'.format(mb_1), 'ZIGaP-EB-{}'.format(mb_2), 'PCA']
	assert len(U_list) == len(title_list)
	n_results = len(U_list)
	fig = plt.figure(figsize=(16, 4))
	s = 30
	alpha = 0.7
	for i in range(len(U_list)):
	    ax = plt.subplot(1, n_results, i+1)
	    handlers = []
	    for c in range(C):
	        h = ax.scatter(U_list[title_list.index(sorted_scores[i][0])][c_train==c, 0], U_list[title_list.index(sorted_scores[i][0])][c_train==c, 1], 
	                       s=s, alpha=alpha, label=clusters_train[c])
	        handlers.append(h)
	    plt.title(sorted_scores[i][0])
	ax.legend(bbox_to_anchor=[1., 1.], frameon=True)
	plt.savefig('{0}/{1}'.format(newpath, tsne_filename), bbox_inches='tight')
	print('Done.\n')

	# Finish script
	elapsed_sec = time.time() - init
	m, s = divmod(elapsed_sec, 60)
	h, m = divmod(m, 60)
	print('Script finished. Time elapsed: {0:.0f}h{1:.0f}m{2:.0f}s.'.format(h, m, s))


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.stderr.write("\nUser interrupt!\n")
		sys.exit(-1)