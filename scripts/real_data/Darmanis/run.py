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
from sklearn.model_selection import train_test_split

import pandas as pd

import time
import operator
import sys, os
import argparse

def main(args):	
	init = time.time()

	newpath = './output'
	if not os.path.exists(newpath):
		print('Creating output directory...')
		os.makedirs(newpath)
		print('Done.\n')

	data_path = '../../../../data/Darmanis/'
	output_filename = 'output.txt'
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
	T = args.n_hours * args.n_minutes * 60.
	if args.n_minutes != 1.:
		T = args.n_minutes * 60
	if args.n_hours != 1.:
		T = args.n_hours * 60 * 60
	Tm, Ts = divmod(T, 60)
	Th, Tm = divmod(Tm, 60)
	S = args.sampling_rate
	max_iter = args.n_iterations
	K = 10 # latent space dimensionality
	param_dict = {'T': '{0:.0f}h{1:.0f}m{2:.0f}s'.format(Th, Tm, Ts), 'S' : S, 'max_iter': max_iter, 'K': K}

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
	pca = ModelWrapper(np.log(X_train + 1.), c_train, name='PCA')
	pca.run(obj.fit_transform)
	print('Done.\n')

	# List of models
	print('Running PCMF models...')
	model1 = pcmf.PCMF(X_train, c_train, X_test=X_test, minibatch_size=args.minibatch_size, sparsity=False)
	model_list = [model1]

	for model in model_list:
		model.run(max_iter=max_iter, max_time=T, sampling_rate=S, do_imp=False, do_holl=False, do_dll=False, verbose=True)
	print('Done.\n')

	# Results
	print('Saving convergence curves to {0}/{1}...'.format(newpath, convergence_filename))
	fig = plt.figure(figsize=(12, 4))
	ax = plt.subplot(1, 2, 1)
	plot_utils.plot_model_convergence(model_list, mode='ll_time', ax=ax, ylabel='Average log-likelihood', xlabel='Seconds(*{0})'.format(S))
	ax = plt.subplot(1, 2, 2)
	plot_utils.plot_model_convergence(model_list, mode='silh_time', ax=ax, ylabel='Silhouette of latent space', xlabel='Seconds(*{0})'.format(S))
	plt.legend(loc='upper left', bbox_to_anchor=[1., 1.], frameon=True)
	plt.suptitle('Zeisel data', fontsize=14)
	plt.subplots_adjust(top=0.85)
	plt.savefig('{0}/{1}'.format(newpath, convergence_filename), bbox_inches='tight')
	print('Done.\n')

	# print_utils.print_model_lls(model_list, mode='Train', filename='{0}/{1}'.format(newpath, output_filename), filemode='a')

	# print_utils.print_model_lls(model_list, mode='Test', filename='{0}/{1}'.format(newpath, output_filename), filemode='a')

	print_utils.print_model_silhouettes(model_list + [pca], filename='{0}/{1}'.format(newpath, output_filename), filemode='a')

	print('Saving 2D projection scatter plots to {0}/{1}...'.format(newpath, tsne_filename))
	fig = plt.figure(figsize=(10, 4))
	ax = plt.axes()
	plot_utils.plot_sorted_tsnes(model_list + [pca], c_train, ax=ax, labels=clusters_train, legend=True)
	plt.savefig('{0}/{1}'.format(newpath, tsne_filename), bbox_inches='tight')
	print('Done.\n')

	# Finish script
	elapsed_sec = time.time() - init
	m, s = divmod(elapsed_sec, 60)
	h, m = divmod(m, 60)
	with open('{0}/{1}'.format(newpath, output_filename), 'a') as f:
		print('Script finished. Time elapsed: {0:.0f}h{1:.0f}m{2:.0f}s.'.format(h, m, s), file=f)
	print('Script finished. Time elapsed: {0:.0f}h{1:.0f}m{2:.0f}s.'.format(h, m, s))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Run Stochastic Variational Inference of PCMF on the Zeisel data set.')
	parser.add_argument('--minibatch_size', type=int, help='number of subsamples to estimate the gradient from', default=1)
	parser.add_argument('--n_hours', metavar='Th', type=float, help='maximum number of hours to run each algorithm', default=1.)
	parser.add_argument('--n_minutes', metavar='Tm', type=float, help='maximum number of minutes to run each algorithm', default=1.)
	parser.add_argument('--sampling_rate', metavar='S', type=float, help='sampling rate of log-likelihoods', default=1)
	parser.add_argument('--n_iterations', metavar='It', type=int, help='maximum number of iterations to run each algorithm', default=1000000)
	parser.add_argument('--n_threads', type=int, help='maximum number of threads to use', default=1)

	args = parser.parse_args()

	os.environ['OPENBLAS_NUM_THREADS'] = str(args.n_threads)

	try:
		main(args)
	except KeyboardInterrupt:
		sys.stderr.write("\nUser interrupt!\n")
		sys.exit(-1)