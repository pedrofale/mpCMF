"""
This file runs 10 CAVI routines for pCMF on 6 different data sets. For each run, the silhouette score
is computed. The 10 results are used to plot a boxplot.
"""

from pCMF.misc import utils, plot_utils
from pCMF.models.pcmf.pcmf import PCMF

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
sns.set_style('whitegrid')

import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from sklearn.metrics import silhouette_score

import operator
import sys, os
import argparse
import json

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", "-f", type=str, required=True, help='JSON containing the simulation settings')
	args = parser.parse_args()
	
	with open(args.file) as f:
		settings = json.load(f)
	
	newpath = './output'
	if not os.path.exists(newpath):
	    os.makedirs(newpath)

	num_zi_levels = settings['num_ZI_levels']
	num_sep_levels = settings['num_separability_levels']
	num_runs = settings['num_runs']

	pca_scores = np.zeros((num_zi_levels, num_sep_levels, num_runs))
	spca_scores = np.zeros((num_zi_levels, num_sep_levels, num_runs))
	pcmf_scores = np.zeros((num_zi_levels, num_sep_levels, num_runs))

	zi_titles = []
	for i in range(num_zi_levels):
		zi_titles.append(list(settings['experiments'][i].keys())[0])

	sep_titles = []
	for i in range(num_sep_levels):
		sep_titles.append(list(settings['experiments'][0][zi_titles[0]][i].keys())[0])

	for i in range(num_zi_levels):
		print('{0}:'.format(zi_titles[i]))
		for j in range(num_sep_levels):
			print('\t{0}'.format(sep_titles[j]))
			exp = settings['experiments'][i][zi_titles[i]][j][sep_titles[j]]

			for k in range(num_runs):
				# Generate data set
				Y, D, X, R, V, U, clusters = utils.generate_data(exp['N'], exp['P'], exp['K'], C=exp['C'], zero_prob=exp['z_p'], eps=exp['eps_U'], return_all=True)
				log_Y = np.log(1 + Y)

				# Run PCA
				pca = PCA(n_components=exp['K']).fit_transform(log_Y)

				# Run SPCA
				spca = SparsePCA(n_components=exp['K']).fit_transform(log_Y)

				# Run CAVI for pCMF
				cavi = PCMF(Y, n_components=exp['K'], sampling_rate=settings['options']['S'], max_time=settings['options']['T'], verbose=True)
				cavi.infer(n_iterations=1000000)

				# Calculate silhouette scores method
				pca_silh = silhouette_score(pca, clusters)
				spca_silh = silhouette_score(spca, clusters)
				pcmf_silh = silhouette_score(cavi.est_U, clusters)

				# Store scores in array
				pca_scores[i, j, k] = pca_silh
				spca_scores[i, j, k] = spca_silh
				pcmf_scores[i, j, k] = pcmf_silh
		print('')

	# Prepare arrays to plot
	no_zi_low_sep = [pca_scores[0, 0, :], spca_scores[0, 0, :], pcmf_scores[0, 0, :]]
	no_zi_high_sep = [pca_scores[0, 1, :], spca_scores[0, 1, :], pcmf_scores[0, 1, :]]

	low_zi_low_sep = [pca_scores[1, 0, :], spca_scores[1, 0, :], pcmf_scores[1, 0, :]]
	low_zi_high_sep = [pca_scores[1, 1, :], spca_scores[1, 1, :], pcmf_scores[1, 1, :]]

	high_zi_low_sep = [pca_scores[2, 0, :], spca_scores[2, 0, :], pcmf_scores[2, 0, :]]
	high_zi_high_sep = [pca_scores[2, 1, :], spca_scores[2, 1, :], pcmf_scores[2, 1, :]]

	print('Plotting boxplots of silhouette scores...')

	# Plot silhouette scores
	legend=['PCA', 'SPCA', 'pCMF']
	fig = plt.figure(figsize=(15, 4))

	ax1 = plt.subplot(1, 3, 1)
	plot_utils.plot_simulation_results(no_zi_low_sep, no_zi_high_sep, ax=ax1, ylabel='Silhouette score', title='No ZI')

	ax2 = plt.subplot(1, 3, 2, sharey=ax1)
	plot_utils.plot_simulation_results(low_zi_low_sep, low_zi_high_sep, ax=ax2, title='Low ZI')

	ax3 = plt.subplot(1, 3, 3, sharey=ax1)
	_, handles = plot_utils.plot_simulation_results(high_zi_low_sep, high_zi_high_sep, ax=ax3, legend=legend, title='High ZI')

	plt.legend(handles, legend, labelspacing=0.5, bbox_to_anchor=(1.3, 0.9))
	_ = [h.set_visible(False) for h in handles]

	plt.savefig('{0}/boxplots.png'.format(newpath))
	print('Done.')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\nUser interrupt!\n")
        sys.exit(-1)
