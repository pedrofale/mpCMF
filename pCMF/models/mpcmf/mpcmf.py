import numpy as np
from pCMF.models.mpcmf.inferences import cavi, cavi_batch, svi, svi_batch
from pCMF.misc import utils

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

class mpCMF(object):
	def __init__(self, Y_train, c_train=None, b_train=None, D_train=None, X_train=None, Y_test=None, c_test=None, b_test=None, n_components=10, empirical_bayes=True, nb=True,
					minibatch_size=None, alpha=None, beta=None, pi_D=None, pi_S=None, zero_inflation=True, sparsity=True, scalings=True, batch_correction=True, do_imp=False, 
					name='m-pCMF', mode='klqp'):
		self.mode = mode
		if self.mode not in ["gibbs", "klqp"]:
			print("Mode \'{}\' unrecognized. Setting to \'klqp\'".format(self.mode))

		self.Y_train = Y_train
		self.c_train = c_train

		self.b_train = b_train # assume one-hot encoded
		self.D_train = D_train
		self.X_train = X_train

		self.do_imp = do_imp
		if do_imp:
			self.corruption_info = utils.dropout(self.Y_train)

		self.dropout_idx = None
		if D_train is not None:
			self.dropout_idx = np.where(self.D_train == 0)

		self.Y_test = Y_test
		self.c_test = c_test
		self.b_test = b_test # assume one-hot encoded

		self.batches = b_train is not None and batch_correction
		if self.b_train is not None:
			self.n_batches = b_train.shape[1] # N x n_batches

		if Y_test is not None:
			assert Y_train.shape[1] == Y_test.shape[1]

		self.P = Y_train.shape[1]

		self.minibatch_size = minibatch_size
		self.empirical_bayes = empirical_bayes

		self.K = n_components

		self.alpha = alpha
		self.beta = beta
		self.pi_D = pi_D
		self.pi_S = pi_S

		if alpha is None:
			self.alpha = 16. * np.ones((2, self.Y_train.shape[0], self.K))
			self.alpha[1] = 4. * np.ones((self.Y_train.shape[0], self.K))
			# self.alpha = np.abs(np.ones((2, self.K)) + np.random.rand(2, self.K))
			# self.alpha[0, :] = 0.1
		if beta is None:
			self.beta = .1 * np.ones((2, self.P, self.K))
			self.beta[1] = .3 * np.ones((self.P, self.K))
			# self.beta = np.abs(np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K))
			# self.beta[0, :, :] = 0.1
			if self.batches:
				self.beta = np.ones((2, self.P, self.K + self.n_batches))
				self.beta[0] = .1 * np.ones((self.P, self.K + self.n_batches))
				self.beta[1] = .3 * np.ones((self.P, self.K + self.n_batches))
		if pi_D is None and zero_inflation:
			zero_probs = np.sum(Y_train==0, axis=0) / Y_train.shape[0]
			self.pi_D = 1.-zero_probs
			self.pi_D[self.pi_D == 1.] = 1. - 1e-7
			self.pi_D[self.pi_D == 0.] = 1e-7
		if pi_S is None and sparsity:
			logit_pi_S = np.random.rand(self.P)
			self.pi_S = np.exp(logit_pi_S) / (1. + np.exp(logit_pi_S))

		if scalings:
			library_size = np.sum(Y_train, axis=1)
			mu_lib = np.mean(library_size)
			var_lib = np.var(library_size)
		else:
			mu_lib = None
			var_lib = None

		self.est_U = None
		self.est_V = None
		self.est_D = None
		self.est_S = None
		self.est_norm_R = None
		self.est_R = None
		self.est_L = None
		self.proj_2d = None
		self.train_ll = None
		self.test_ll = None
		self.silhouette = None
		self.asw = None
		self.ari = None
		self.nmi = None
		self.dropid_acc = None
		self.dropimp_err = None
		self.batch_asw = None

		self.name = name
		if self.name is None:
			if self.mode is "klqp":
				if minibatch_size is None:
					if empirical_bayes:
						self.name = "CAVI-EB"
					else:
						self.name = "CAVI"
				else:
					if empirical_bayes:
						self.name = "SVI-EB-{}".format(minibatch_size)
					else:
						self.name = "SVI-{}".format(minibatch_size)
			else:
				if empirical_bayes:
					self.name = "Gibbs-EB"
				else:
					self.name = "Gibbs"

		self.inf = None
		print("{}:".format(self.name))

		if minibatch_size is None:
			if self.batches:
				self.inf = cavi_batch.CoordinateAscentVI(Y_train, self.alpha, self.beta, b_train=b_train, b_test=b_test, X_test=self.Y_test, pi_D=self.pi_D, pi_S=self.pi_S, mu_lib=mu_lib, var_lib=var_lib, empirical_bayes=empirical_bayes)
			else:
				self.inf = cavi.CoordinateAscentVI(Y_train, self.alpha, self.beta, X_test=self.Y_test, pi_D=self.pi_D, pi_S=self.pi_S, mu_lib=mu_lib, var_lib=var_lib, empirical_bayes=empirical_bayes, nb=nb)
		else:
			if self.batches:
				self.inf = svi_batch.StochasticVI(Y_train, self.alpha, self.beta, b_train=b_train, b_test=b_test, X_test=self.Y_test, pi_D=self.pi_D, pi_S=self.pi_S, mu_lib=mu_lib, var_lib=var_lib, minibatch_size=minibatch_size, empirical_bayes=empirical_bayes)
			else:
				self.inf = svi.StochasticVI(Y_train, self.alpha, self.beta, X_test=self.Y_test, pi_D=self.pi_D, pi_S=self.pi_S, mu_lib=mu_lib, var_lib=var_lib, minibatch_size=minibatch_size, empirical_bayes=empirical_bayes)

	def run(self, max_iter=10, min_iter=1, tol=None, sampling_rate=1, max_time=60, calc_elbo=True, calc_test=False, calc_silh=False, 
		do_tsne=True, do_dll=True, do_holl=True, do_silh=True, do_imp=True, do_batch=False, verbose=False):
		if verbose:
			print('Running {0}...'.format(self.name))

		X = self.Y_train
		if self.do_imp:
			X = self.corruption_info['X_corr']
			
		if self.minibatch_size is not None:
			# in the case we want SVI, the number of iterations (max_iter and min_iter) corresponds to number of epochs
			iterep = int(X.shape[0]/float(self.minibatch_size))
			max_iter = max(iterep * max_iter, 1)
			min_iter = max(iterep * min_iter, 1)

		self.inf.run(max_iter=max_iter, min_iter=min_iter, tol=tol,
			calc_elbo=calc_elbo, calc_test=calc_test, calc_silh=calc_silh, clusters=self.c_train, sampling_rate=sampling_rate, max_time=max_time)
		ests = self.inf.get_estimates()
		self.est_U = ests['U']
		self.est_V = ests['V']
		self.est_D = ests['D']
		self.est_S = ests['S']
		self.est_L = ests['L']
		est_U = self.est_U

		if self.batches:
			est_U = np.concatenate((est_U, self.b_train), axis=1) # N x (K + n_batches)
		self.est_norm_R = np.dot(est_U, self.est_V.T)
		self.est_R = self.est_L[:, np.newaxis] * self.est_norm_R

		if self.do_imp:
			est_mean = self.est_R
			X_original = self.Y_train

			self.dropimp_err = utils.imputation_error(est_mean, X_original, 
				self.corruption_info['X_corr'], self.corruption_info['i'], self.corruption_info['j'], self.corruption_info['ix'])	
		else:
			if self.K > 2:
				if do_tsne:
					self.do_tsne()
			else:
				self.proj_2d = self.est_U

			if do_dll:
				if verbose:
					print('Evaluating train-data log-likelihood...')
				if self.inf.__class__.__name__ == 'StochasticVI':
					self.train_ll = self.inf.compute_elbo(self.Y_train, n_iterations=min(50, max_iter))
				else:
					if self.batches:
						self.train_ll = self.inf.compute_elbo()
					else:
						self.train_ll = self.inf.compute_elbo()
			
			if do_holl:
				if verbose:
					print('Evaluating test-data log-likelihood...')
				if self.Y_test is not None:
					if self.batches:
						assert self.b_test is not None
					self.test_ll = self.inf.compute_elbo(self.Y_test, n_iterations=min(50, max_iter))

			if do_silh and self.c_train is not None:
				self.silhouette = silhouette_score(self.est_U, self.c_train)
				self.asw = self.silhouette
				if self.batches or do_batch:
					if self.n_batches > 2:
						raise ValueError("Only 2 batches supported.")
					self.batch_asw = silhouette_score(self.est_U, self.b_train[:, 0]) # this only works for 2 batches!!!

				C = np.unique(self.c_train).size
				kmeans = KMeans(n_clusters=C, n_init=200, n_jobs=8)
				res = kmeans.fit_predict(self.est_U)

				self.ari = adjusted_rand_score(self.c_train, res)

				self.nmi = normalized_mutual_info_score(self.c_train, res)

		if verbose:
			print('Done.')

	def do_tsne(self):
		self.proj_2d = TSNE(n_components=2).fit_transform(self.est_U)

	def do_holl(self):
		if self.Y_test is not None:
			if self.batches:
				assert self.b_test is not None
			self.test_ll = self.inf.predictive_ll()

	def sample_posterior(self, return_all=False):
		U = utils.sample_gamma(self.inf.a[0], self.inf.a[1])
		V = utils.sample_gamma(self.inf.b[0], self.inf.b[1])
		S = utils.sample_bernoulli(self.inf.p_S)

		R = np.matmul(U, V.T)
		L = utils.sample_gamma(self.inf.n[0], self.inf.n[1])
		R = R * L[:, np.newaxis]
		X = np.random.poisson(R)

		D = utils.sample_bernoulli(p=self.inf.p_D)
		Y = np.where(D == 0, np.zeros((self.Y_train.shape)), X)

		dict_all = {'U': U, 'V': V, 'S': S, 'R': R, 'X': X, 'Y': Y, 'D': D}
		return dict_all