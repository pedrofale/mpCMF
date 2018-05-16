import numpy as np
from pCMF.models.pcmf import cavi, svi
from pCMF.misc import utils

class PCMF(object):
	def __init__(self, train_data, n_components=10, sampling_rate=1, max_time=60, alpha=None, beta=None, pi_D=None, pi_S=None, verbose=True):
		self.train_data = train_data

		P = train_data.shape[1]

		self.inferences = ['gibbs', 'cavi', 'svi']

		self.n_components = n_components
		self.sampling_rate = sampling_rate
		self.max_time = max_time

		self.verbose = verbose

		self.alpha = alpha
		self.beta = beta
		self.pi_D = pi_D

		if alpha is None:
			self.alpha = np.ones((2, n_components))
			self.alpha[0, :] = 3.
			self.alpha[1, :] = 0.5
		if beta is None:
			self.beta = np.ones((2, P, n_components))
		if pi_D is None:
			self.pi_D = np.ones((P,)) * 0.5

		self.inf = None
		self.ll = None
		self.est_U = None
		self.est_V = None

	def infer(self, test_data=None, algorithm='cavi', empirical_bayes=True, minibatch_size=1, n_iterations=100):
		if algorithm in self.inferences:
			if algorithm is 'cavi':
				# Run CAVI and get estimates
				self.inf = cavi.CoordinateAscentVI(self.train_data, self.alpha, self.beta, self.pi_D)
				self.ll = self.inf.run_cavi(X_test=test_data, n_iterations=n_iterations, empirical_bayes=empirical_bayes, return_ll=True, 
					sampling_rate=self.sampling_rate, max_time=self.max_time, verbose=self.verbose)
				self.est_U = self.inf.a[0] / self.inf.a[1] # VI estimate is the mean of the variational approximation
				self.est_V = self.inf.b[0] / self.inf.b[1] # VI estimate is the mean of the variational approximation

			if algorithm is 'svi':
				# Run SVI and get estimates
				self.inf = svi.StochasticVI(self.train_data, self.alpha, self.beta, self.pi_D)
				self.ll = self.inf.run_svi(X_test=test_data, n_iterations=n_iterations, empirical_bayes=empirical_bayes, minibatch_size=minibatch_size, return_ll=True, 
					sampling_rate=self.sampling_rate, max_time=self.max_time, verbose=self.verbose)
				self.est_U = self.inf.a[0] / self.inf.a[1] # VI estimate is the mean of the variational approximation
				self.est_V = self.inf.b[0] / self.inf.b[1] # VI estimate is the mean of the variational approximation
		else:
			print('Inference algorithm unrecognized.')

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