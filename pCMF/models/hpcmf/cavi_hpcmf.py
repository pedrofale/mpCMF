""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Hierarchical Probabilistic Count Matrix 
Factorization model.

This draws inspiration from Gopalan et al, 2013, where they develop a Hierarchical Poisson Factorization
model for recommendation systems. The difference here is that we also account for zero-inflation.

This model imposes sparsity in the component loadings through the hierarchical prior structure on V,
instead of introducing a sparsity variable as in pCMF.

With regard to pCMF, this is an alternative to doing Empirical Bayes estimation of the hyperparameters.
For small data, the Hierarchical Bayes approach should perform better.
"""

import time
import math
import numpy as np
from scipy.special import digamma, factorial
from pCMF.misc.utils import log_likelihood

class CoordinateAscentVI(object):
	def __init__(self, X, alpha, beta, pi):
		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.shape[1] # latent space dim
		
		# Hyperparameters		
		self.a_ = 0.3 * np.ones((self.N))
		self.b_ = 1. * np.ones((self.N))
		self.c_ = 0.3 * np.ones((self.P))
		self.d_ = 1. * np.ones((self.P))

		self.alpha = 0.3 * np.ones((self.N, self.K)) # a
		self.beta = 0.3 * np.ones((self.P, self.K)) # c

		self.pi = np.expand_dims(pi, axis=0).repeat(self.N, axis=0) # NxP
		self.logit_pi = np.log(self.pi / (1. - self.pi))

		# Variational parameters
		self.a = np.ones((2, self.N, self.K)) + np.random.rand(2, self.N, self.K)# parameters of q(U)
		self.b = np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K) # parameters of q(V)
		self.r = np.ones((self.N, self.P, self.K)) * 0.5 # parameters of q(Z)
		self.p = np.ones((self.N, self.P)) * 0.5 # parameters of q(D)

		self.k = np.ones((2, self.N)) # parameters of q(psi)
		self.k[0] = self.a_ + self.K * self.alpha[:, 0]
		self.t = np.ones((2, self.P)) # parameters of q(mu)
		self.t[0] = self.c_ + self.K * self.beta[:, 0]

	def estimate_U(self, a):
		return a[0] / a[1]

	def estimate_V(self, b):
		return b[0] / b[1]

	def update_k(self, k, a):
		# k is 2xN, a is 2xNxK
		k[1] = self.a_ / self.b_ + np.sum(self.a[0] / self.a[1], axis=1) # sum over K
		return k

	def update_t(self):
		# t is 2xP, b is 2xPxK
		self.t[1] = self.c_ / self.d_ + np.sum(self.b[0] / self.b[1], axis=1)

	def update_a(self, a, X, p, r, k_):
		N = X.shape[0]

		for i in range(N):
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for j in range(self.P):
					total1 = total1 + p[i, j] * X[i, j] * r[i, j, k]
					total2 = total2 + p[i, j] * self.b[0, j, k] / self.b[1, j, k]
				a[0, i, k] = self.alpha[i, k] + total1
				a[1, i, k] = k_[0, i] / k_[1, i] + total2

		return a

	def update_b(self):
		for j in range(self.P):
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for i in range(self.N):
					total1 = total1 + self.p[i, j] * self.X[i, j] * self.r[i, j, k]
					total2 = total2 + self.p[i, j] * self.a[0, i, k] / self.a[1, i, k]
				self.b[0, j, k] = self.beta[j, k] + total1
				self.b[1, j, k] = self.t[0, j] / self.t[1, j] + total2

	def update_p(self, p, X, a, r):
		N = X.shape[0]

		logit_p = np.zeros((N, self.P))
		for i in range(N):
			for j in range(self.P):
				logit_p[i, j] = self.logit_pi[i, j] - np.sum(a[0, i, :]/a[1, i, :] * self.b[0, j, :]/self.b[1, j, :])
		p = np.exp(logit_p) / (1. + np.exp(logit_p))
		p[X != 0] = 1.

		return p

	def update_r(self, r, X, a, p):
		N = X.shape[0]

		ar = digamma(a[0]) - np.log(a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		aux = np.zeros((self.K,))	
		for i in range(N):
			for j in range(self.P):
				aux = np.exp(ar[i, :] + br[j, :])
				r[i, j,:] = aux / np.sum(aux)

		return r

	def run_cavi(self, X_test=None, n_iterations=10, return_ll=True, sampling_rate=10, max_time=60, verbose=True):
		""" Run coordinate ascent variational inference and return 
		variational parameters.
		
		Get the log-likelihood every sampling_rate seconds.
		"""
		if return_ll:			
			ll_it = []
			ll_time = []

		# init clock
		start = time.time()
		init = start
		for it in range(n_iterations):
			# update the local variables
			self.a = self.update_a(self.a, self.X, self.p, self.r, self.k)
			self.p = self.update_p(self.p, self.X, self.a, self.r)
			self.r = self.update_r(self.r, self.X, self.a, self.p)
			self.k = self.update_k(self.k, self.a) # hyperparameter

			# update global variables
			self.update_b()
			self.update_t() # hyperparameter
			
			if return_ll:
				# compute the LL
				if X_test is not None:
					ll_curr = self.predictive_ll(X_test)
				else:
					# subsample the data to evaluate the ll in
					idx = np.random.randint(self.N, size=100)
					est_U = self.estimate_U(self.a[:, idx, :])
					est_V = self.estimate_V(self.b)

					ll_curr = log_likelihood(self.X[idx], est_U, est_V, self.p[idx])
				end = time.time()
				it_time = end - start
				if it_time >= sampling_rate - 0.1*sampling_rate:
					ll_time.append(ll_curr)
					start = end
				ll_it.append(ll_curr)
				if verbose:
					if X_test is not None:
						print("Iteration {0}/{1}. Held-out log-likelihood: {2:.3f}. Elapsed: {3:.0f} seconds".format(it, n_iterations, ll_curr, end-init), end="\r")
					else:
						print("Iteration {0}/{1}. Log-likelihood: {2:.3f}. Elapsed: {3:.0f} seconds".format(it, n_iterations, ll_curr, end-init), end="\r")
				if (end - init) >= max_time:
					break
			elif verbose:
				print("Iteration {}/{}".format(it+1, n_iterations), end="\r")	
		if return_ll: 
			return ll_it, ll_time

