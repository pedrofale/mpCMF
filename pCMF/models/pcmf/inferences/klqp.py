""" This file contains an abstract class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model.
"""

import time
import numpy as np
from sklearn.metrics import silhouette_score
from pCMF.models.pcmf.utils import log_likelihood

from abc import ABC, abstractmethod

class KLqp(ABC):
	def __init__(self, X, alpha, beta, pi, empirical_bayes=False):
		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.shape[1] # latent space dim

		# Empirical Bayes estimation of hyperparameters
		self.empirical_bayes = empirical_bayes

		# Hyperparameters		
		self.alpha = np.expand_dims(alpha, axis=1).repeat(self.N, axis=1) # 2xNxK
		self.beta = beta # 2xPxK
		self.pi =  np.expand_dims(pi, axis=0).repeat(self.N, axis=0) # NxP
		self.logit_pi = np.log(self.pi / (1. - self.pi))

		# Variational parameters
		self.a = np.ones((2, self.N, self.K)) + np.random.rand(2, self.N, self.K) # parameters of q(U)
		self.b = np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K) # parameters of q(V)
		self.r = np.random.dirichlet([1 / self.K] * self.K, size=(self.N, self.P,))
		#self.r = np.ones((self.N, self.P, self.K)) * 0.5 # parameters of q(Z)
		self.p = np.ones((self.N, self.P)) * 0.5 # parameters of q(D)

		# Log-likelihood per iteration and per time unit
		self.ll_it = []
		self.ll_time = []

		# Silhouette per iteration and per time unit
		self.silh_it = []
		self.silh_time = []

	def estimate_U(self, a):
		return a[0] / a[1]

	def estimate_V(self, b):
		return b[0] / b[1]

	def estimate_D(self, p):
		D = np.zeros((self.N, self.P))
		D[p_S > 0.5] = 1.
		return S

	def predictive_ll(self, X_test, n_iterations=10, S=100):
		""" Computes the average posterior predictive likelihood of data not used for 
		training: p(X_test | X). It uses the posterior parameter estimates and 
		Monte Carlo sampling to compute the integrals using S samples.
		"""
		N_test = X_test.shape[0]

		a = np.ones((2, N_test, self.K)) + np.random.rand(2, N_test, self.K)# parameters of q(U) for test data
		r = np.ones((N_test, self.P, self.K)) * 0.5 # parameters of q(Z) for test data
		p = np.ones((N_test, self.P)) * 0.5 # parameters of q(D) for test data

		# Posterior approximation over the local latent variables for each
		# of the new data points.
		for i in range(n_iterations):
			r = self.update_r(r, X_test, a, p) # parameter of Multinomial
			a = self.update_a(a, X_test, p, r) # parameters of Gamma
			p = self.update_p(p, X_test, a, r) # parameters of Bernoulli
		
		# # Monte Carlo estimation of the posterior predictive: use S samples
		# U_samples = sample_gamma(a[0], a[1], size=S) # S by X_test.shape[0] by K
		# D_samples = sample_bernoulli(p, size=S) # S by X_test.shape[0] by P
		# V_samples = sample_gamma(self.b[0], self.b[1], size=S) # SxPxK

		est_U = self.estimate_U(a)
		est_V = self.estimate_V(self.b)

		pred_ll = log_likelihood(X_test, est_U, est_V, p) # S
		pred_ll = np.mean(pred_ll)
			
		return pred_ll

	def generate_from_posterior(return_all=False):
		U = utils.sample_gamma(self.a[0], self.a[1])
		V = utils.sample_gamma(self.b[0], self.b[1])

		R = np.matmul(U.T, V)
		X = np.random.poisson(R)

		D = utils.sample_bernoulli(p=self.p)
		Y = np.where(D == 1, np.zeros((self.N, self.P)), X)

		if return_all:
			return Y, U, V, D
		return Y

	@abstractmethod
	def update_parameters(self):
		pass

	def run(self, n_iterations=10, calc_ll=False, calc_silh=False, max_time=60, sampling_rate=1., clusters=None):
		if calc_silh:
			if clusters is None:
				print("Can't compute silhouette score without cluster assignments.")
				calc_silh = False

		# init clock
		start = time.time()
		init = start
		for it in range(n_iterations):
			self.update_parameters(it)

			# Update log-likelihood and silhouette
			if calc_ll:
				# Subsample the data to evaluate the ll in
				idx = np.random.randint(self.N, size=np.min([100, self.N]))
				est_U = self.estimate_U(self.a[:, idx, :])
				est_V = self.estimate_V(self.b)
				ll_curr = log_likelihood(self.X[idx], est_U, est_V, self.p[idx])
				self.ll_it.append(ll_curr)
			
			if calc_silh:
				est_U = self.estimate_U(self.a)
				silh_curr = silhouette_score(est_U, clusters)
				self.silh_it.append(silh_curr)

			end = time.time()
			it_time = end - start
			
			if it_time >= sampling_rate - 0.1*sampling_rate:
				if calc_ll:
					self.ll_time.append(ll_curr)
				if calc_silh:
					self.silh_time.append(silh_curr)
				start = end

			# Print status
			elapsed = end-init
			if calc_ll:
				print("Iteration {0}/{1}. Log-likelihood: {2:.3f}. Elapsed: {3:.0f} seconds".format(it, n_iterations, ll_curr, elapsed), end="\r")
			else:
				print("Iteration {0}/{1}. Elapsed: {2:.0f} seconds".format(it+1, n_iterations, elapsed), end="\r")

			# If maximum run time has passed, stop
			if (end - init) >= max_time:
				break

		print('')
