from pCMF.misc import utils, plot_utils, print_utils
from pCMF.misc.model_wrapper import ModelWrapper
from pCMF.models.pcmf import pcmf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import seaborn as sns
sns.set_style('whitegrid')

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from scipy.stats import gamma

import argparse
import os, sys
import time

def main(args):
	init = time.time()

	newpath = './output'
	if not os.path.exists(newpath):
		print('Creating output directory...')
		os.makedirs(newpath)
		print('Done.\n')

	output_filename = 'output.txt'
	convergence_filename = 'convergence.png'
	tsne_filename = 'tsne.png'
	dropimp_filename = 'dropout_imputation.png'

	# Experiment parameters
	N = args.n_samples # number of observations
	P = args.n_features # observation space dimensionality
	K = args.n_components # latent space dimensionality
	C = args.n_clusters # number of clusters

	# Generate data set
	z_p = args.dropout_prob
	eps = args.cluster_separability
	Y, D, X, R, V, U, clusters = utils.generate_sparse_data(N, P, K, C=C, zero_prob=z_p, noisy_prop=0.5,
	                                             eps_U=eps, return_all=True)

	Y_train, Y_test, X_train, X_test, D_train, D_test, U_train, U_test, c_train, c_test = train_test_split(Y, 
			X, D, U.T, clusters, test_size=0.2, random_state=42)

	# Experiment parameters
	T = args.n_minutes * 60.
	Tm, Ts = divmod(T, 60)
	Th, Tm = divmod(Tm, 60)
	S = args.sampling_rate
	max_iter = args.n_iterations
	param_dict = {'N': N, 'P': P, 'K': K, 'T': '{0:.0f}h{1:.0f}m{2:.0f}s'.format(Th, Tm, Ts), 'S' : S, 'max_iter': max_iter}
	
	with open('{0}/{1}'.format(newpath, output_filename), 'w') as f:
		print('Experiment parameters:', file=f)
		for param in param_dict:
			print('{0}: {1}'.format(param, param_dict[param]), file=f)
		print('', file=f)

	print('Experiment parameters:')
	for param in param_dict:
		print('{0}: {1}'.format(param, param_dict[param]))
	print('')

	# Run PCA
	print('Running PCA...')
	obj = PCA(n_components=K)
	pca = ModelWrapper(np.log(Y_train + 1.), c_train, D_train=D_train, name='PCA')
	pca.run(obj.fit_transform)
	print('Done.\n')

	# List of models
	print('Running PCMF models...')
	model1 = pcmf.PCMF(Y_train, c_train, X_test=Y_test, D_train=D_train, minibatch_size=1)
	model2 = pcmf.PCMF(Y_train, c_train, X_test=Y_test, D_train=D_train, minibatch_size=10)
	model3 = pcmf.PCMF(Y_train, c_train, X_test=Y_test, D_train=D_train, minibatch_size=100)
	model4 = pcmf.PCMF(Y_train, c_train, X_test=Y_test, D_train=D_train, minibatch_size=500)
	model5 = pcmf.PCMF(Y_train, c_train, X_test=Y_test, D_train=D_train)
	model_list = [model1, model2, model3, model4, model5]

	for model in model_list:
		model.run(max_iter=max_iter, max_time=T, sampling_rate=S, verbose=True)
	print('Done.\n')

	print('Saving convergence curves to {0}/{1}...'.format(newpath, convergence_filename))
	fig = plt.figure(figsize=(12, 4))
	ax = plt.subplot(1, 2, 1)
	plot_utils.plot_model_convergence(model_list, mode='ll_time', ax=ax, ylabel='Average log-likelihood', xlabel='Seconds(*{0})'.format(S))
	ax = plt.subplot(1, 2, 2)
	plot_utils.plot_model_convergence(model_list, mode='silh_time', ax=ax, ylabel='Silhouette of latent space', xlabel='Seconds(*{0})'.format(S))
	plt.legend(loc='upper left', bbox_to_anchor=[1., 1.], frameon=True)
	plt.suptitle('Data set with N={} and P={}'.format(N, P), fontsize=14)
	plt.subplots_adjust(top=0.85)
	plt.savefig('{0}/{1}'.format(newpath, convergence_filename), bbox_inches='tight')
	print('Done.\n')

	print_utils.print_model_lls(model_list, mode='Train', filename='{0}/{1}'.format(newpath, output_filename), filemode='a')

	print_utils.print_model_lls(model_list, mode='Test', filename='{0}/{1}'.format(newpath, output_filename), filemode='a')

	print_utils.print_model_silhouettes(model_list + [pca], filename='{0}/{1}'.format(newpath, output_filename), filemode='a')

	print('Saving 2D projection scatter plots to {0}/{1}...'.format(newpath, tsne_filename))
	fig = plt.figure(figsize=(14, 9))
	ax = plt.axes()
	plot_utils.plot_sorted_tsnes(model_list + [pca], c_train, ax=ax, legend=False, nrows=2, ncols=3)
	plt.savefig('{0}/{1}'.format(newpath, tsne_filename), bbox_inches='tight')
	print('Done.\n')

	print_utils.print_model_dropid_acc(model_list, filename='{0}/{1}'.format(newpath, output_filename), filemode='a')
	print_utils.print_model_dropimp_err(model_list, filename='{0}/{1}'.format(newpath, output_filename), filemode='a')

	print('Saving dropout imputation density plots to {0}/{1}...'.format(newpath, dropimp_filename))
	fig = plt.figure(figsize=(14, 9))
	ax = plt.axes()
	plot_utils.plot_sorted_imputation_densities(model_list, X_train, ax=ax, ymax=50, nrows=2, ncols=3)
	plt.savefig('{0}/{1}'.format(newpath, dropimp_filename), bbox_inches='tight')
	print('Done.\n')

	# Finish script
	elapsed_sec = time.time() - init
	m, s = divmod(elapsed_sec, 60)
	h, m = divmod(m, 60)
	with open('{0}/{1}'.format(newpath, output_filename), 'a') as f:
		print('Script finished. Time elapsed: {0:.0f}h{1:.0f}m{2:.0f}s.'.format(h, m, s), file=f)
	print('Script finished. Time elapsed: {0:.0f}h{1:.0f}m{2:.0f}s.'.format(h, m, s))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Compare minibatch sizes in Stochastic Variational Inference.')
	parser.add_argument('n_samples', metavar='N', type=int, help='number of samples')
	parser.add_argument('n_features', metavar='P', type=int, help='number of features')
	parser.add_argument('n_components', metavar='K', type=int, help='number of components')
	parser.add_argument('--n_clusters', metavar='C', type=int, help='number of clusters', default=2)
	parser.add_argument('--n_minutes', metavar='T', type=float, help='maximum number of minutes to run each algorithm', default=1.)
	parser.add_argument('--sampling_rate', metavar='S', type=float, help='sampling rate of log-likelihoods', default=1)
	parser.add_argument('--n_iterations', metavar='It', type=int, help='maximum number of iterations to run each algorithm', default=1000000)
	parser.add_argument('--dropout_prob', metavar='z_p', type=float, help='probability of dropouts', default=0.5)
	parser.add_argument('--cluster_separability', metavar='eps', type=float, help='degree of separability of clusters in latent space', default=5.)
	parser.add_argument('--n_threads', type=int, help='maximum number of threads to use', default=1)

	args = parser.parse_args()

	os.environ['OPENBLAS_NUM_THREADS'] = str(args.n_threads)

	try:
		main(args)
	except KeyboardInterrupt:
		sys.stderr.write("\nUser interrupt!\n")
		sys.exit(-1)