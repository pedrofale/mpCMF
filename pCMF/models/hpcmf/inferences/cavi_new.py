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

import numpy as np
from scipy.special import digamma, factorial
from pCMF.models.hpcmf.inferences.klqp import KLqp

class CoordinateAscentVI(KLqp):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

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

	def update_k(self, k, a):
		# k is 2xN, a is 2xNxK
		k[1] = self.a_ / self.b_ + np.sum(self.a[0] / self.a[1], axis=1) # sum over K
		return k

	def update_t(self):
		# t is 2xP, b is 2xPxK
		self.t[1] = self.c_ / self.d_ + np.sum(self.b[0] / self.b[1], axis=1)

	def update_parameters(self, *args):
		# update the local variables
		self.a = self.update_a(self.a, self.X, self.p, self.r, self.k)
		if self.zi:
			self.p = self.update_p(self.p, self.X, self.a, self.r)
		self.r = self.update_r(self.r, self.X, self.a, self.p)
		self.k = self.update_k(self.k, self.a) # hyperparameter

		# update global variables
		self.update_b()
		self.update_t() # hyperparameter