""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Hierarchical Probabilistic Count Matrix 
Factorization model.

This draws inspiration from Gopalan et al, 2013, where they develop a Hierarchical Poisson Factorization
model for recommendation systems. The difference here is that we also account for zero-inflation.

This model imposes sparsity in the component loadings through the hierarchical prior structure on V,
instead of introducing a sparsity variable as in pCMF.

With regard to pCMF, this is an alternative to doing Empirical Bayes estimation of the hyperparameters.
For small data, the Hierarchical Bayes approach should perform better.

Here we use Stochastic Variational Inference.
"""

import numpy as np
from scipy.special import digamma, factorial
from pCMF.models.hpcmf.inferences.klqp_new import KLqp

class StochasticVI(KLqp):
	def __init__(self, *args, minibatch_size=1, delay=1, forget_rate=0.9, **kwargs):
		super().__init__(*args, **kwargs)

		self.minibatch_size = minibatch_size
		self.delay = delay
		self.forget_rate = forget_rate

		if minibatch_size > self.N:
			print("Warning: minibatch size can't be larger than the number of samples.")
			print("Setting minibatch size to 1.")
			self.minibatch_size = 1

def update_k(self, k, a, mb_idx=None):
		# k is 2xN, a is 2xNxK
		k[1, mb_idx] = self.a_[mb_idx] / self.b_[mb_idx] + np.sum(self.a[0, mb_idx, :] / self.a[1, mb_idx, :], axis=1) # sum over K
		return k

	def update_t(self):
		# t is 2xP, b is 2xPxK
		self.t[1] = self.c_ / self.d_ + np.sum(self.b[0] / self.b[1], axis=1)

	def update_a(self, a, X, p, r, k_, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			iterator = range(N)
		else:
			iterator = mb_idx

		for i in iterator:
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for j in range(self.P):
					total1 = total1 + p[i, j] * X[i, j] * r[i, j, k]
					total2 = total2 + p[i, j] * self.b[0, j, k] / self.b[1, j, k]
				a[0, i, k] = self.alpha[i, k] + total1
				a[1, i, k] = k_[0, i] / k_[1, i] + total2

		return a

	def update_b(self, minibatch_indexes, eta):
		S = minibatch_indexes.size
		intermediate_b = np.ones((S, 2, self.P, self.K))

		for s in range(S):
			i = minibatch_indexes[s]
			for j in range(self.P):
				for k in range(self.K):
					intermediate_b[s, 0, j, k] = self.beta[j, k] + self.N * self.p[i, j] * self.X[i, j] * self.r[i, j, k]
					intermediate_b[s, 1, j, k] = self.t[0, j] + self.N * self.p[i, j] * self.a[0, i, k] / self.a[1, i, k]
	
		self.b = (1.-eta)*self.b + eta*np.mean(intermediate_b, axis=0)

	def update_p(self, p, X, a, r, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			iterator = range(N)
		else:
			iterator = mb_idx

		logit_p = np.zeros((self.P,))
		for i in iterator:
			for j in range(self.P):
				logit_p[j] = self.logit_pi[i, j] - np.sum(a[0, i, :]/a[1, i, :] * self.b[0, j, :]/self.b[1, j, :])
		
		if mb_idx is None:
			p[:, :] = np.exp(logit_p) / (1. + np.exp(logit_p))		
		else:
			p[mb_idx, :] = np.exp(logit_p) / (1. + np.exp(logit_p))
		p[X != 0] = 1.

		return p

	def update_r(self, r, X, a, p, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			iterator = range(N)
		else:
			iterator = mb_idx

		ar = digamma(a[0]) - np.log(a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		aux = np.zeros((self.K,))	
		for i in iterator:
			for j in range(self.P):
				aux = np.exp(ar[i, :] + br[j, :])
				r[i, j,:] = aux / np.sum(aux)

		return r

	def update_parameters(self, it):
		# sample data point uniformly from the data set
		mb_idx = np.random.randint(self.N, size=minibatch_size)

		# update the local variables
		self.a = self.update_a(self.a, self.X, self.p, self.r, self.k, mb_idx=mb_idx)
		if self.zi:
			self.p = self.update_p(self.p, self.X, self.a, self.r, mb_idx=mb_idx)
		self.r = self.update_r(self.r, self.X, self.a, self.p, mb_idx=mb_idx)
		self.k = self.update_k(self.k, self.a, mb_idx=mb_idx) # hyperparameter

		# update global variables
		step_size = (it+1. + delay)**(-forget_rate)
		self.update_b(mb_idx, step_size)
		self.update_t() # hyperparameter