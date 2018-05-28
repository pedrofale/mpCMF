"""
This file compares CAV and SV inference of the non-sparse pCMF on simulated data sets.
We generate data sets with 2 cell types with different levels of separability and 
zero-inflation. 

Here we focus on comparing the running time of both inference algorithms. Does SVI achieve
the same likelihood as CAVI? We limit the running time to 2 hours per experiment. 

INPUT
The file receives as input the data set dimensions, N and P. We always reduce to K=10.

OUTPUT
We output the average log likelihood curves per second and iteration, the 2D TSNE plots
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy.stats import gamma

import operator
import sys, os
import argparse
import json

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", "-f", type=str, required=True, help='JSON containing the simulation settings')
	args = parser.parse_args()
	
	with open(args.file) as f:
    	data = json.load(f)
	
	newpath = './output'
	if not os.path.exists(newpath):
	    os.makedirs(newpath)


	exp = utils.load_experiment('')

	# Run PCA
	pca = PCA(n_components=exp['K']).fit_transform(exp['Y_train'])

	# Run CAVI for pCMF
	cavi = PCMF(train_data=exp['Y_train'], n_components=exp['K'], empirical_bayes=True, sampling_rate=exp['S'], max_time=exp['T'], inf='cavi')

	# Run SVI for pCMF
	svi = PCMF(train_data=exp['Y_train'], n_components=exp['K'], minibatch_size=exp['mb'], empirical_bayes=True, sampling_rate=exp['S'], max_time=exp['T'], inf='svi')		

	# Output results
	newpath = './output/{}'.format(exp.exp_name)
	print('Saving results to {}.'.format(newpath))
	if not os.path.exists(newpath):
	    os.makedirs(newpath)

	# Plot convergence curves
	print('Plotting convergence curves...')
	utils.plot_convergence_curves(title='Data set with N={} and P={}'.format(exp['N_train'], exp['P']), curves=[cavi_ll[0], svi_ll[0]], 
		labels=['CAVI', 'SVI-{0}'.format(exp['mb'])], xlabel='Iterations', file='{0}/convergence_iterations.png'.format(newpath))
	utils.plot_convergence_curves(title='Data set with N={} and P={}'.format(exp['N_train'], exp['P']), curves=[cavi_ll[1], svi_ll[1]],
		labels=['CAVI', 'SVI-{0}'.format(exp['mb'])], xlabel='Seconds', file='{0}/convergence_seconds.png'.format(newpath))

	# Compute and plot 2D TSNE scatter plots
	print('Computing 2D TSNE projections...')
	pca_tsne = TSNE(n_components=2).fit_transform(pca)
	cavi_tsne = TSNE(n_components=2).fit_transform(cavi.est_U)
	svi_tsne = TSNE(n_components=2).fit_transform(svi.est_U)

	print('Plotting 2D TSNE scatter plots...')
	U_list = [pca_TSNE, cavi_TSNE, svi_TSNE]
	title_list = ['PCA', 'CAVI', 'SVI']
	utils.plot_scatter_tsne_projections(U_list, title_list, n_clusters=exp['C'], cluster_annonations=exp['c_train'], file='{0}/2D_scatter.png'.format(newpath))
	
	# Compute and print out silhouette scores to file
	print('Computing silhouette scores on the latent space...')
	latent_list = [pca, cavi.est_U, svi.est_U]
	names_list = ['PCA', 'CAVI', 'SVI']
	silhouette_list = utils.compute_silhouette_scores(latent_list)
	utils.print_silhouette_scores(silhouette_list, names_list, file='{0}/silhouette_scores.txt'.format(newpath))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\nUser interrupt!\n")
        sys.exit(-1)