""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model.

Here we use Stochastic Variational Inference.
"""

import numpy as np
from scipy.special import digamma, factorial
from pCMF.models.pcmf.klqp import KLqp
from pCMF.misc.utils import psi_inverse

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

	def update_a(self, a, X, p, r, mb_idx=None):
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
				a[0, i, k] = self.alpha[0, i, k] + total1
				a[1, i, k] = self.alpha[1, i, k] + total2

		return a

	def update_p(self, p, X, a, r, mb_idx=None):
		N = X.shape[0]	
		if mb_idx is None:
			iterator = range(N)
		else:
			iterator = mb_idx

		n = len(iterator)
		
		logit_p = np.zeros((N, self.P))
		for i in iterator:
			for j in range(self.P):
				logit_p[i, j] = self.logit_pi[i, j] - np.sum(a[0, i, :]/a[1, i, :] * self.b[0, j, :]/self.b[1, j, :])
		
		if mb_idx is None:
			p[:, :] = np.exp(logit_p) / (1. + np.exp(logit_p))		
		else:
			p[mb_idx, :] = np.exp(logit_p[mb_idx, :]) / (1. + np.exp(logit_p[mb_idx, :]))
		p[X != 0] = 1. - 1e-7
		p[p == 1.] = 1 - 1e-7
		p[p == 0.] = 1e-7

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
				#for k in range(self.K):
				#	aux[k] = np.exp(a[i, k] + b[j, k])
				r[i,j,:] = aux / np.sum(aux)

		return r

	def update_b(self, minibatch_indexes, eta):
		S = minibatch_indexes.size
		intermediate_b = np.ones((S, 2, self.P, self.K))

		for s in range(S):
			i = minibatch_indexes[s]
			for j in range(self.P):
				for k in range(self.K):
					intermediate_b[s, 0, j, k] = self.beta[0, j, k] + self.N * self.p[i, j] * self.X[i, j] * self.r[i, j, k]
					intermediate_b[s, 1, j, k] = self.beta[1, j, k] + self.N * self.p[i, j] * self.a[0, i, k] / self.a[1, i, k]

		self.b = (1.-eta)*self.b + eta*np.mean(intermediate_b, axis=0)

	def update_pi(self, minibatch_indexes):
		""" Empirical Bayes update of the hyperparameter pi
		"""
		# pi is NxP
		pi = np.mean(self.p[minibatch_indexes], axis=0)
		
		self.pi = np.expand_dims(pi, axis=0).repeat(self.N, axis=0)
		self.logit_pi = np.log(self.pi) - np.log(1. - self.pi)

	def update_alpha(self, minibatch_indexes):
		""" Empirical Bayes update of the hyperparameter alpha
		"""
		# alpha1 and alpha2 are NxK
		S = minibatch_indexes.size

		self.alpha[0, minibatch_indexes] = np.log(self.alpha[1, minibatch_indexes]) + np.expand_dims(np.mean(digamma(self.a[0, minibatch_indexes]) - np.log(self.a[1, minibatch_indexes]), axis=0), axis=0).repeat(S, axis=0)
		alpha_1 = self.alpha[0, minibatch_indexes[0], :]

		for k in range(self.K):
			alpha_1[k] = psi_inverse(2., self.alpha[0, minibatch_indexes[0], k])

		self.alpha[0] = np.expand_dims(alpha_1, axis=0).repeat(self.N, axis=0)
		self.alpha[1] = self.alpha[0] / np.mean(self.a[0, minibatch_indexes] / self.a[1, minibatch_indexes], axis=0)

	def update_beta(self, minibatch_indexes):
		""" Empirical Bayes update of the hyperparameter beta
		"""
		self.beta[0] = np.log(self.beta[1]) + np.expand_dims(np.mean(digamma(self.b[0]) - np.log(self.b[1]), axis=0), axis=0).repeat(self.P, axis=0)
		beta_1 = self.beta[0, 0, :]

		for k in range(self.K):
			beta_1[k] = psi_inverse(2., self.beta[0, 0, k])

		self.beta[0] = np.expand_dims(beta_1, axis=0).repeat(self.P, axis=0)
		self.beta[1] = self.beta[0] / np.mean(self.b[0] / self.b[1], axis=0)

	def update_parameters(self, it):
		# sample data point uniformly from the data set
		mb_idx = np.random.randint(self.N, size=self.minibatch_size)

		# update the local variables corresponding to the sampled data point
		self.a = self.update_a(self.a, self.X, self.p, self.r, mb_idx=mb_idx)
		self.p = self.update_p(self.p, self.X, self.a, self.r, mb_idx=mb_idx)
		self.r = self.update_r(self.r, self.X, self.a, self.p, mb_idx=mb_idx)

		# update global variables, considering an hypothetical data set 
		# containing N replicates of sample n and a new step_size
		step_size = (it+1. + self.delay)**(-self.forget_rate)
		self.update_b(mb_idx, step_size)

		if self.empirical_bayes:
			# update hyperparameters
			self.update_pi(mb_idx)
			self.update_alpha(mb_idx)
			self.update_beta(mb_idx)
