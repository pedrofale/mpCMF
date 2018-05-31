""" This file contains an abstract class for inference of the parameters of the variational
distributions that approximate the posteriors of the Hierarchical Probabilistic Count Matrix Factorization
model.
"""

import time
import numpy as np
from scipy.special import factorial
from sklearn.metrics import silhouette_score
from pCMF.misc.utils import log_likelihood

from abc import ABC, abstractmethod

class KLqp(ABC):
	def __init__(self, X, alpha, beta, pi=None):
		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.shape[1] # latent space dim

		self.zi = pi_D is not None

		# If counts are big, clip the log-likelihood to avoid inf values
		self.clip_ll = np.any(np.isinf(factorial(self.X)))

		## Hyperparameters
		self.a_ = 0.3 * np.ones((self.N))
		self.b_ = 1. * np.ones((self.N))
		self.c_ = 0.3 * np.ones((self.P))
		self.d_ = 1. * np.ones((self.P))

		self.alpha = 0.3 * np.ones((self.N, self.K)) # a
		self.beta = 0.3 * np.ones((self.P, self.K)) # c
		
		# zero-Inflation
		if self.zi:
			print('Considering zero-inflated counts.')
			self.pi_D =  np.expand_dims(pi_D, axis=0).repeat(self.N, axis=0) # NxP
			self.logit_pi_D = np.log(self.pi_D / (1. - self.pi_D))
		else:
			self.pi_D =  np.ones((self.N, self.P)) # NxP
		
		# sparsity
		if self.sparse:
			print('Considering loading sparsity.')
			self.pi_S =  np.expand_dims(pi_S, axis=1).repeat(self.K, axis=1) # PxK
			self.logit_pi_S = np.log(self.pi_S / (1. - self.pi_S))
		else:
			self.pi_S =  np.ones((self.P, self.K)) # PxK

		## Variational parameters
		self.a = np.ones((2, self.N, self.K)) + np.random.rand(2, self.N, self.K)# parameters of q(U)
		self.b = np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K) # parameters of q(V)
		self.r = np.ones((self.N, self.P, self.K)) * 0.5 # parameters of q(Z)

		self.k = np.ones((2, self.N)) # parameters of q(psi)
		self.k[0] = self.a_ + self.K * self.alpha[:, 0]
		self.t = np.ones((2, self.P)) # parameters of q(mu)
		self.t[0] = self.c_ + self.K * self.beta[:, 0]

		# zero-inflation
		self.p = np.ones((self.N, self.P)) # parameters of q(D)
		if self.zi:
			 self.p = self.p * 0.5

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

	def estimate_D(self, p_D, thres=0.5):
		D = np.zeros((self.N, self.P))
		D[p_D > thres] = 1.
		return D

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

		pred_ll = log_likelihood(X_test, est_U, est_V, p_D, clip=self.clip_ll)
		pred_ll = np.mean(pred_ll)
			
		return pred_ll

	def generate_from_posterior(return_all=False):
		U = utils.sample_gamma(self.a[0], self.a[1])
		V = utils.sample_gamma(self.b[0], self.b[1])

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

				ll_curr = log_likelihood(self.X[idx], est_U, est_V, self.p_D[idx], clip=self.clip_ll)
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
			m, s = divmod(elapsed, 60)
			h, m = divmod(m, 60)
			if calc_ll:
				print("Iteration {0}/{1}. Log-likelihood: {2:.3f}. Elapsed: {3:.0f}h{4:.0f}m{5:.0f}s".format(it, n_iterations, ll_curr, h, m, s), end="\r")
			else:
				print("Iteration {0}/{1}. Elapsed: {2:.0f}h{3:.0f}m{4:.0f}s".format(it+1, n_iterations, h, m, s), end="\r")

			# If maximum run time has passed, stop
			if elapsed >= max_time:
				break

		print('')
