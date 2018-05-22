""" This file contains an abstract class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model.
"""

import time
import numpy as np
from sklearn.metrics import silhouette_score
from pCMF.misc.utils import log_likelihood

from abc import ABC, abstractmethod

class KLqp(ABC):
	def __init__(self, X, alpha, beta, pi_D=None, pi_S=None, empirical_bayes=False):
		np.random.seed(42)

		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.shape[1] # latent space dim

		self.zi = pi_D is not None
		self.sparse = pi_S is not None

		# Empirical Bayes estimation of hyperparameters
		self.empirical_bayes = empirical_bayes

		## Hyperparameters		
		self.alpha = np.expand_dims(alpha, axis=1).repeat(self.N, axis=1) # 2xNxK
		self.beta = beta # 2xPxK
		
		# zero-Inflation
		if self.zi:
			self.pi_D =  np.expand_dims(pi_D, axis=0).repeat(self.N, axis=0) # NxP
			self.logit_pi_D = np.log(self.pi_D / (1. - self.pi_D))
		else:
			self.pi_D =  np.ones((self.N, self.P)) # NxP
		
		# sparsity
		if self.sparse:
			self.pi_S =  np.expand_dims(pi_S, axis=1).repeat(self.K, axis=1) # PxK
			self.logit_pi_S = np.log(self.pi_S / (1. - self.pi_S))
		else:
			self.pi_S =  np.ones((self.P, self.K)) # PxK

		## Variational parameters
		self.a = np.ones((2, self.N, self.K)) + np.random.rand(2, self.N, self.K) # parameters of q(U)
		self.b = np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K) # parameters of q(V)
		self.r = np.random.dirichlet([1 / self.K] * self.K, size=(self.N, self.P,)) # parameters of q(Z)
		#self.r = np.ones((self.N, self.P, self.K)) * 0.5 # parameters of q(Z)
		
		# zero-inflation
		self.p_D = np.ones((self.N, self.P)) # parameters of q(D)
		if self.zi:
			 self.p_D = self.p_D * 0.5

		# sparsity
		self.p_S = np.ones((self.P, self.K)) # parameters of q(S)
		if self.sparse:
			 self.p_S = self.p_S * 0.5

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

	def estimate_D(self, p_D):
		D = np.zeros((self.N, self.P))
		D[p_D > 0.5] = 1.
		return D

	def estimate_S(self, p_S, tresh=0.5):
		S = np.zeros((self.P, self.K))
		S[p_S > tresh] = 1.
		return S

	def predictive_ll(self, X_test, n_iterations=10, S=100):
		""" Computes the average posterior predictive likelihood of data not used for 
		training: p(X_test | X). It uses the posterior parameter estimates and 
		Monte Carlo sampling to compute the integrals using S samples.
		"""
		N_test = X_test.shape[0]

		a = np.ones((2, N_test, self.K)) + np.random.rand(2, N_test, self.K)# parameters of q(U) for test data
		r = np.ones((N_test, self.P, self.K)) * 0.5 # parameters of q(Z) for test data
		p_D = np.ones((N_test, self.P)) # parameters of q(D) for test data
		if self.zi:
			p_D = p_D * 0.5

		# Posterior approximation over the local latent variables for each
		# of the new data points.
		for i in range(n_iterations):
			r = self.update_r(r, X_test, a, p_D) # parameter of Multinomial
			a = self.update_a(a, X_test, p_D, r) # parameters of Gamma
			if self.zi:
				p_D = self.update_p_D(p_D, X_test, a, r) # parameters of Bernoulli
		
		# # Monte Carlo estimation of the posterior predictive: use S samples
		# U_samples = sample_gamma(a[0], a[1], size=S) # S by X_test.shape[0] by K
		# D_samples = sample_bernoulli(p, size=S) # S by X_test.shape[0] by P
		# V_samples = sample_gamma(self.b[0], self.b[1], size=S) # SxPxK

		est_U = self.estimate_U(a)
		est_V = self.estimate_V(self.b)
		est_S = self.estimate_S(self.p_S)

		pred_ll = log_likelihood(X_test, est_U, est_V, p_D, est_S) # S
		pred_ll = np.mean(pred_ll)
			
		return pred_ll

	def generate_from_posterior(return_all=False):
		U = utils.sample_gamma(self.a[0], self.a[1])
		V = utils.sample_gamma(self.b[0], self.b[1])

		if self.sparse:
			est_S = self.estimate_S(self.p_S)
			V = V * est_S

		R = np.matmul(U.T, V)
		X = np.random.poisson(R)

		if self.zi:
			D = utils.sample_bernoulli(p=self.p_D)
			Y = np.where(D == 1, np.zeros((self.N, self.P)), X)
		else:
			Y = X

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
				est_S = self.estimate_S(self.p_S)

				ll_curr = log_likelihood(self.X[idx], est_U, est_V, self.p_D[idx], est_S)
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
