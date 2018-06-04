import numpy as np
from pCMF.models.pcmf.inferences import cavi_new, svi_new
from pCMF.misc import utils

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, accuracy_score

class PCMF(object):
	def __init__(self, Y_train, c_train, D_train=None, X_train=None, Y_test=None, c_test=None, n_components=10, empirical_bayes=True, 
					minibatch_size=None, alpha=None, beta=None, pi_D=None, pi_S=None, zero_inflation=True, sparsity=True, name=None):
		self.Y_train = Y_train
		self.c_train = c_train

		self.D_train = D_train
		self.X_train = X_train

		self.dropout_idx = None
		if D_train is not None:
			self.dropout_idx = np.where(self.D_train == 0)

		self.Y_test = Y_test
		self.c_test = c_test

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
			self.alpha = np.abs(np.ones((2, self.K)) + np.random.rand(2, self.K))
		if beta is None:
			self.beta = np.abs(np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K))
		if pi_D is None and zero_inflation:
			logit_pi_D = np.random.rand(self.P)
			self.pi_D = np.exp(logit_pi_D) / (1. + np.exp(logit_pi_D))
		if pi_S is None and sparsity:
			logit_pi_S = np.random.rand(self.P)
			self.pi_S = np.exp(logit_pi_S) / (1. + np.exp(logit_pi_S))

		self.est_U = None
		self.est_V = None
		self.est_D = None
		self.est_S = None
		self.est_R = None
		self.proj_2d = None
		self.train_ll = None
		self.test_ll = None
		self.silhouette = None
		self.dropid_acc = None
		self.dropimp_err = None

		self.name = name
		if self.name is None:
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

		self.inf = None
		print("{}:".format(self.name))
		if minibatch_size is None:
			self.inf = cavi_new.CoordinateAscentVI(Y_train, self.alpha, self.beta, pi_D=self.pi_D, pi_S=self.pi_S, empirical_bayes=empirical_bayes)
		else:
			self.inf = svi_new.StochasticVI(Y_train, self.alpha, self.beta, pi_D=self.pi_D, pi_S=self.pi_S, minibatch_size=minibatch_size, empirical_bayes=empirical_bayes)

	def run(self, max_iter=1, sampling_rate=1, max_time=60, calc_ll=True, calc_silh=True, do_tsne=True, do_dll=True, do_holl=True, do_silh=True, do_imp=True, verbose=False):
		if verbose:
			print('Running {0}...'.format(self.name))
		self.inf.run(n_iterations=max_iter, calc_ll=calc_ll, calc_silh=calc_silh, clusters=self.c_train, sampling_rate=sampling_rate, max_time=max_time)
		
		self.est_U = self.inf.a[0] / self.inf.a[1] # VI estimate is the mean of the variational approximation
		self.est_V = self.inf.b[0] / self.inf.b[1]
		self.est_D = self.inf.estimate_D(self.inf.p_D)
		self.est_S = self.inf.estimate_S(self.inf.p_S)
		self.est_R = np.dot(self.est_U, self.est_V.T)

		if do_tsne:
			self.proj_2d = TSNE(n_components=2).fit_transform(self.est_U)

		if do_dll:
			self.train_ll = utils.log_likelihood(self.Y_train, self.est_U, self.est_V, self.inf.p_D, self.est_S, clip=self.inf.clip_ll)
		
		if do_holl:
			if self.Y_test is not None:
				self.test_ll = self.inf.predictive_ll(self.Y_test)

		if do_silh:
			self.silhouette = silhouette_score(self.est_U, self.c_train)

		if do_imp:
			if self.D_train is not None:
				self.dropid_acc = accuracy_score(self.est_D.flatten(), self.D_train.flatten())
				if self.X_train is not None:
					self.dropimp_err = utils.imputation_error(self.X_train, self.est_R, self.dropout_idx)

		if verbose:
			print('Done.')

	def sample_posterior(return_all=False):
		U = utils.sample_gamma(self.inf.a[0], self.inf.a[1])
		V = utils.sample_gamma(self.inf.b[0], self.inf.b[1])

		R = np.matmul(U.T, V)
		X = np.random.poisson(R)

		D = utils.sample_bernoulli(p=self.inf.p)
		Y = np.where(D == 1, np.zeros((N, P)), X)

		if return_all:
			return Y, U, V, D
		return Y