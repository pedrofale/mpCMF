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
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy.stats import gamma

import time

import operator
import sys, os
import argparse
import json

def main():	
	init = time.time()

	newpath = './output'
	if not os.path.exists(newpath):
		print('Creating output directory...')
		os.makedirs(newpath)
		print('Done.')

	# Experiment parameters
	N = 100 # number of observations
	P = 1000 # observation space dimensionality
	K = 10 # latent space dimensionality
	C = 2 # number of clusters

	T = 60. * 5.
	S = 10.

	z_p_arr = [0., 0.3, 0.6] # none, low, high ZI
	eps_arr = [2., 10.] # low, high separability
	n_runs = 10 # number of experiments per setting

	# Prior parameters
	alpha = np.ones((2, K))
	alpha[0, :] = 3.
	alpha[1, :] = 0.5
	beta = np.ones((2, P, K))
	pi_D = np.ones((P,)) * 0.5

	zi_debug_strs = ['NO zero-inflation', 'LOW zero-inflation', 'HIGH zero-inflation']
	eps_debug_strs = ['LOW separability', 'HIGH separability']

	silh_pca_scores = np.zeros((len(z_p_arr), len(eps_arr), n_runs))
	silh_gap_scores = np.zeros((len(z_p_arr), len(eps_arr), n_runs))
	silh_zigap_scores = np.zeros((len(z_p_arr), len(eps_arr), n_runs))

	dll_gap_scores = np.zeros((len(z_p_arr), len(eps_arr), n_runs))
	dll_zigap_scores = np.zeros((len(z_p_arr), len(eps_arr), n_runs))

	holl_gap_scores = np.zeros((len(z_p_arr), len(eps_arr), n_runs))
	holl_zigap_scores = np.zeros((len(z_p_arr), len(eps_arr), n_runs))

	for i in range(len(z_p_arr)):
		print('%s:' % zi_debug_strs[i])
		for j in range(len(eps_arr)):
			print('\t%s' % eps_debug_strs[j])
			for k in range(n_runs):
				# Generate data set
				Y, D, X, R, V, U, clusters = utils.generate_data(N, P, K, C=C, zero_prob=z_p_arr[i], 
				                                             eps=eps_arr[j], return_all=True)
				Y_train, Y_test, U_train, U_test, c_train, c_test = train_test_split(Y, U.T, clusters, test_size=0.2, random_state=42)

				log_Y = np.log(1 + Y_train)

				# Run PCA
				pca = PCA(n_components=2).fit_transform(log_Y)

				# Run GaP
				print('GaP:')
				infgap = cavi_new.CoordinateAscentVI(Y_train, alpha, beta, empirical_bayes=True)
				infgap.run(n_iterations=1000000, calc_ll=True, calc_silh=True, clusters=c_train, sampling_rate=S, max_time=T)
				gap_U = infgap.a[0] / infgap.a[1] # VI estimate is the mean of the variational approximation
				gap_V = infgap.b[0] / infgap.b[1]
				gap_S = infgap.estimate_S(infgap.p_S)

				# Run ZIGaP
				print('ZIGaP:')
				infzigap = cavi_new.CoordinateAscentVI(Y_train, alpha, beta, pi_D=pi_D, empirical_bayes=True)
				infzigap.run(n_iterations=1000000, calc_ll=True, calc_silh=True, clusters=c_train, sampling_rate=S, max_time=T)
				zigap_U = infzigap.a[0] / infzigap.a[1] # VI estimate is the mean of the variational approximation
				zigap_V = infzigap.b[0] / infzigap.b[1]
				zigap_S = infzigap.estimate_S(infzigap.p_S)

				# Calculate silhouette scores
				pca_silh = silhouette_score(pca, c_train)
				gap_silh = silhouette_score(gap_U, c_train)
				zigap_silh = silhouette_score(zigap_U, c_train)

				# Store silhs in array
				silh_pca_scores[i, j, k] = pca_silh
				silh_gap_scores[i, j, k] = gap_silh
				silh_zigap_scores[i, j, k] = zigap_silh

				# Calculate train data log-likelihood
				gap_dll = utils.log_likelihood(Y_train, gap_U, gap_V, infgap.p_D, gap_S)
				zigap_dll = utils.log_likelihood(Y_train, zigap_U, zigap_V, infzigap.p_D, zigap_S)

				# Store dlls in array
				dll_gap_scores[i, j, k] = gap_dll
				dll_zigap_scores[i, j, k] = zigap_dll

				# Calculate train data log-likelihood
				gap_holl = infgap.predictive_ll(Y_test)
				zigap_holl = infzigap.predictive_ll(Y_test)

				# Store holls in array
				holl_gap_scores[i, j, k] = gap_holl
				holl_zigap_scores[i, j, k] = zigap_holl

	print('')

	print('Saving results to {0}...'.format(newpath))

	np.save('{0}/silh_pca_scores.npy'.format(newpath), silh_pca_scores)
	np.save('{0}/silh_gap_scores.npy'.format(newpath), silh_gap_scores)
	np.save('{0}/silh_zigap_scores.npy'.format(newpath), silh_zigap_scores)

	np.save('{0}/dll_gap_scores.npy'.format(newpath), dll_gap_scores)
	np.save('{0}/dll_zigap_scores.npy'.format(newpath), dll_zigap_scores)

	np.save('{0}/holl_gap_scores.npy'.format(newpath), holl_gap_scores)
	np.save('{0}/holl_zigap_scores.npy'.format(newpath), holl_zigap_scores)

	print('Done.')

	elapsed = time.time() - init
	print('\nScript finished in {0:.0f} seconds.'.format(elapsed))

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.stderr.write("\nUser interrupt!\n")
		sys.exit(-1)