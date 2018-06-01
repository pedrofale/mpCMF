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

	newpath = args.output_dir
	if not os.path.exists(newpath):
		print('Creating output directory...')
		os.makedirs(newpath)
	print('Done.\n')

	output_filename = 'output.txt'
	convergence_filename = 'convergence.png'

	# Experiment parameters
	N = args.n_samples # number of observations
	P = args.n_features # observation space dimensionality
	K = args.n_components # latent space dimensionality
	C = args.n_clusters # number of clusters

	z_p = args.dropout_prob

	noisy_prop_arr = [0., 0.4, 0.7] # none, low, high noise
	eps_arr = [2., 10.] # low, high separability

	# Experiment parameters
	T = args.n_minutes * 60.
	Tm, Ts = divmod(T, 60)
	Th, Tm = divmod(Tm, 60)
	S = args.sampling_rate
	max_iter = args.n_iterations
	param_dict = {'N': N, 'P': P, 'K': K, 'T': '{0:.0f}h{1:.0f}m{2:.0f}s'.format(Th, Tm, Ts), 'S' : S, 'max_iter': max_iter, 'z_p': z_p}
	
	with open('{0}/{1}'.format(newpath, output_filename), 'w') as f:
		print('Experiment parameters:', file=f)
		for param in param_dict:
			print('{0}: {1}'.format(param, param_dict[param]), file=f)
		print('', file=f)

	print('Experiment parameters:')
	for param in param_dict:
		print('{0}: {1}'.format(param, param_dict[param]))
	print('')

	noisy_debug_strs = []
	for i in noisy_prop_arr:
		noisy_debug_strs.append('{}% noisy genes'.format(noisy_prop_arr[0]))
	eps_debug_strs = ['LOW separability', 'HIGH separability']

	n_runs = args.n_runs
	silh_pca_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	silh_gap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	silh_zigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	silh_szigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))

	dll_gap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	dll_zigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	dll_szigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))

	holl_gap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	holl_zigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	holl_szigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))

	dropid_zigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	dropid_szigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))

	dropimp_gap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	dropimp_zigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))
	dropimp_szigap_scores = np.zeros((len(noisy_prop_arr), len(eps_arr), n_runs))

	for i in range(len(noisy_prop_arr)):
		print('\n--%s:' % noisy_debug_strs[i])
		for j in range(len(eps_arr)):
			print('\n\t--%s' % eps_debug_strs[j])
			for k in range(args.n_runs):
				# Generate data set
				Y, D, X, R, V, U, clusters = utils.generate_sparse_data(N, P, K, C=C, zero_prob=z_p, noisy_prop=noisy_prop_arr[i],
				                                             eps_U=eps_arr[j], return_all=True)
				Y_train, Y_test, X_train, X_test, D_train, D_test, U_train, U_test, c_train, c_test = train_test_split(Y, X, D, U.T, clusters, test_size=0.2, random_state=42)

				# Run PCA
				print('Running PCA...')
				obj = PCA(n_components=K)
				pca = ModelWrapper(np.log(Y_train + 1.), c_train, D_train=D_train, name='PCA')
				pca.run(obj.fit_transform)
				print('Done.\n')

				# List of models
				print('Running PCMF models...')
				gap = pcmf.PCMF(Y_train, c_train, X_test=Y_test, D_train=D_train, name="GaP", zero_inflation=False, sparsity=False, minibatch_size=args.minibatch_size)
				zigap = pcmf.PCMF(Y_train, c_train, X_test=Y_test, D_train=D_train, name="ZIGaP", zero_inflation=True, sparsity=False, minibatch_size=args.minibatch_size)
				szigap = pcmf.PCMF(Y_train, c_train, X_test=Y_test, D_train=D_train, name="SZIGaP", zero_inflation=True, sparsity=True, minibatch_size=args.minibatch_size)
				model_list = [gap, zigap, szigap]

				for model in model_list:
					model.run(max_iter=max_iter, max_time=T, sampling_rate=S, verbose=True, calc_silh=False, calc_ll=False)
				print('Done.\n')

				# Store silhs in array
				silh_pca_scores[i, j, k] = pca.silhouette
				silh_gap_scores[i, j, k] = gap.silhouette
				silh_zigap_scores[i, j, k] = zigap.silhouette
				silh_szigap_scores[i, j, k] = szigap.silhouette

				# Store dlls in array
				dll_gap_scores[i, j, k] = gap.train_ll
				dll_zigap_scores[i, j, k] = zigap.train_ll
				dll_szigap_scores[i, j, k] = szigap.train_ll

				# Store holls in array
				holl_gap_scores[i, j, k] = gap.test_ll
				holl_zigap_scores[i, j, k] = zigap.test_ll
				holl_szigap_scores[i, j, k] = szigap.test_ll

				# Store dropid in array
				dropid_zigap_scores[i, j, k] = zigap.dropid_acc
				dropid_szigap_scores[i, j, k] = szigap.dropid_acc

				# Store dropimp in array
				dropimp_gap_scores[i, j, k] = gap.dropimp_err
				dropimp_zigap_scores[i, j, k] = zigap.dropimp_err
				dropimp_zigap_scores[i, j, k] = szigap.dropimp_err

				# if i == 0 and j == 0 and k == 0:
				# 	# Store the convergence curves from the first iteration
				# 	print('Saving convergence curves to {0}/{1}...'.format(newpath, convergence_filename))
				# 	fig = plt.figure(figsize=(12, 4))
				# 	ax = plt.subplot(1, 2, 1)
				# 	plot_utils.plot_model_convergence(model_list, mode='ll_it', ax=ax, ylabel='Average log-likelihood', xlabel='Seconds(*{0})'.format(S))
				# 	ax = plt.subplot(1, 2, 2)
				# 	plot_utils.plot_model_convergence(model_list, mode='silh_it', ax=ax, ylabel='Silhouette of latent space', xlabel='Seconds(*{0})'.format(S))
				# 	plt.legend(loc='upper left', bbox_to_anchor=[1., 1.], frameon=True)
				# 	plt.suptitle('Data set with N={} and P={}'.format(N, P), fontsize=14)
				# 	plt.subplots_adjust(top=0.85)
				# 	plt.savefig('{0}/{1}'.format(newpath, convergence_filename), bbox_inches='tight')
				# 	print('Done.\n')

	print('Saving results to {0}...'.format(newpath))

	np.save('{0}/silh_pca_scores.npy'.format(newpath), silh_pca_scores)
	np.save('{0}/silh_gap_scores.npy'.format(newpath), silh_gap_scores)
	np.save('{0}/silh_zigap_scores.npy'.format(newpath), silh_zigap_scores)
	np.save('{0}/silh_szigap_scores.npy'.format(newpath), silh_szigap_scores)

	np.save('{0}/dll_gap_scores.npy'.format(newpath), dll_gap_scores)
	np.save('{0}/dll_zigap_scores.npy'.format(newpath), dll_zigap_scores)
	np.save('{0}/dll_szigap_scores.npy'.format(newpath), dll_szigap_scores)

	np.save('{0}/holl_gap_scores.npy'.format(newpath), holl_gap_scores)
	np.save('{0}/holl_zigap_scores.npy'.format(newpath), holl_zigap_scores)
	np.save('{0}/holl_szigap_scores.npy'.format(newpath), holl_szigap_scores)

	np.save('{0}/dropid_zigap_scores.npy'.format(newpath), dropid_zigap_scores)
	np.save('{0}/dropid_szigap_scores.npy'.format(newpath), dropid_szigap_scores)

	np.save('{0}/dropimp_gap_scores.npy'.format(newpath), dropimp_gap_scores)
	np.save('{0}/dropimp_zigap_scores.npy'.format(newpath), dropimp_zigap_scores)
	np.save('{0}/dropimp_szigap_scores.npy'.format(newpath), dropimp_szigap_scores)

	print('Done.\n')

	# Finish script
	elapsed_sec = time.time() - init
	m, s = divmod(elapsed_sec, 60)
	h, m = divmod(m, 60)
	print('Script finished. Time elapsed: {0:.0f}h{1:.0f}m{2:.0f}s.'.format(h, m, s))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Compare minibatch sizes in Stochastic Variational Inference.')
	parser.add_argument('n_samples', metavar='N', type=int, help='number of samples')
	parser.add_argument('n_features', metavar='P', type=int, help='number of features')
	parser.add_argument('n_components', metavar='K', type=int, help='number of components')
	parser.add_argument('--minibatch_size', help='number of subsamples to estimate the gradient from', default=None)
	parser.add_argument('--n_clusters', metavar='C', type=int, help='number of clusters', default=2)
	parser.add_argument('--n_minutes', metavar='T', type=float, help='maximum number of minutes to run each algorithm', default=1.)
	parser.add_argument('--sampling_rate', metavar='S', type=float, help='sampling rate of log-likelihoods', default=1)
	parser.add_argument('--n_iterations', metavar='It', type=int, help='maximum number of iterations to run each algorithm', default=1000000)
	parser.add_argument('--dropout_prob', metavar='z_p', type=float, help='probability of dropouts', default=0.5)
	parser.add_argument('--n_runs', type=int, help='number of data sets to generate', default=1)
	parser.add_argument('--output_dir', type=str, help='name of directory to store results', default='./output')
	parser.add_argument('--n_threads', type=int, help='maximum number of threads to use', default=1)

	args = parser.parse_args()

	os.environ['OPENBLAS_NUM_THREADS'] = str(args.n_threads)

	if args.minibatch_size is not None:
		args.minibatch_size = int(args.minibatch_size)

	try:
		main(args)
	except KeyboardInterrupt:
		sys.stderr.write("\nUser interrupt!\n")
		sys.exit(-1)