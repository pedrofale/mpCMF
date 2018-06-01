""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model.

The variational parameter updates were all derived from Equation (40) of Blei, D. et al 2016. 
Basically, each parameter of the variational approximation of some latent variable is set as 
the expected value of the natural parameter of that variable's complete conditional.
"""

import numpy as np
from scipy.special import digamma, factorial
from pCMF.models.pcmf.inferences.klqp_new import KLqp
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

	def update_a(self, a, X, p_D, r, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			iterator = range(N)
		else:
			iterator = mb_idx

		for i in iterator:
			for k in range(self.K):
				a[0, i, k] = self.alpha[0, i, k] + np.sum(p_D[i, :] * self.p_S[:, k] * X[i, :] * r[i, :, k])
				a[1, i, k] = self.alpha[1, i, k] + np.sum(p_D[i, :] * self.p_S[:, k] * self.b[0, :, k] / self.b[1, :, k])

		return a

	def update_b(self, minibatch_indexes, eta):
		S = minibatch_indexes.size
		intermediate_b = np.ones((S, 2, self.P, self.K))

		for s in range(S):
			i = minibatch_indexes[s]
			for j in range(self.P):
				for k in range(self.K):
					intermediate_b[s, 0, j, k] = self.beta[0, j, k] + self.p_S[j, k] * self.N * self.p_D[i, j] * self.X[i, j] * self.r[i, j, k]
					intermediate_b[s, 1, j, k] = self.beta[1, j, k] + self.p_S[j, k] * self.N * self.p_D[i, j] * self.a[0, i, k] / self.a[1, i, k]
		
		self.b = (1.-eta)*self.b + eta*np.mean(intermediate_b, axis=0)

	def update_p_D(self, p_D, X, a, r, mb_idx=None):
		N = X.shape[0]

		if mb_idx is None:
			iterator = range(N)
		else:
			iterator = mb_idx

		#logit_p = self.logit_pi - np.matmul(self.a[0]/self.a[1], (self.b[0]/self.b[1].T))
		logit_p_D = np.zeros((N, self.P))
		for i in iterator:
			for j in range(self.P):
				logit_p_D[i, j] = self.logit_pi_D[i, j] - np.sum(self.p_S[j, :] * a[0, i, :]/a[1, i, :] * self.b[0, j, :]/self.b[1, j, :])
		
		if mb_idx is None:
			p_D[:, :] = np.exp(logit_p_D) / (1. + np.exp(logit_p_D))		
		else:
			p_D[mb_idx, :] = np.exp(logit_p_D[mb_idx, :]) / (1. + np.exp(logit_p_D[mb_idx, :]))

		p_D[X != 0] = 1. - 1e-7
		p_D[p_D == 1.] = 1 - 1e-7
		p_D[p_D == 0.] = 1e-7

		return p_D

	def update_r(self, r, X, a, p_D, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			iterator = range(N)
		else:
			iterator = mb_idx

		S = self.estimate_S(self.p_S)

		ar = digamma(a[0]) - np.log(a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		aux = np.zeros((self.K,))	
		for i in iterator:
			for j in range(self.P):
				if np.all(S[j, :] == 0.):
					r[i, j, :] = 0.
				else:
					aux = S[j, :] * np.exp(ar[i, :] + br[j, :])
					r[i, j, :] = aux / np.sum(aux)

		return r

	def update_p_S(self, minibatch_indexes, eta):
		S = minibatch_indexes.size
		intermediate_logit_p_S = np.ones((S, self.P, self.K))

		ar = digamma(self.a[0]) - np.log(self.a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK

		logit_p_S = np.zeros((self.P, self.K))

		for s in range(S):
			i = minibatch_indexes[s]
			for j in range(self.P):
				for k in range(self.K):
					intermediate_logit_p_S[s, j, k] = self.logit_pi_S[j, k] + self.N *(-self.p_D[i, j] * self.a[0, i, k]/self.a[1, i, k] * self.b[0, j, k]/self.b[1, j, k] +
						self.p_D[i, j] * self.r[i, j, k] * self.X[i, j] * (ar[i, k] + br[j, k]))

		logit_p_S = (1.-eta)*self.logit_p_S + eta*np.mean(intermediate_logit_p_S, axis=0)		
		
		self.p_S[logit_p_S > 300] = 1.
		self.p_S[logit_p_S < 300] = np.exp(logit_p_S[logit_p_S < 300]) / (1. + np.exp(logit_p_S[logit_p_S < 300]))
		self.p_S[self.p_S == 0.] = 1e-7
		self.p_S[self.p_S == 1.] = 1 - 1e-7

		self.logit_p_S = np.log(self.p_S / (1. - self.p_S))

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

	def update_beta(self):
		""" Empirical Bayes update of the hyperparameter beta
		"""
		self.beta[0] = np.log(self.beta[1]) + np.expand_dims(np.mean(digamma(self.b[0]) - np.log(self.b[1]), axis=0), axis=0).repeat(self.P, axis=0)
		beta_1 = self.beta[0, 0, :]

		for k in range(self.K):
			beta_1[k] = psi_inverse(2., self.beta[0, 0, k])

		self.beta[0] = np.expand_dims(beta_1, axis=0).repeat(self.P, axis=0)
		self.beta[1] = self.beta[0] / np.mean(self.b[0] / self.b[1], axis=0)

	def update_pi_D(self, minibatch_indexes):
		""" Empirical Bayes update of the hyperparameter pi_D
		"""
		# pi is NxP
		pi = np.mean(self.p_D[minibatch_indexes], axis=0)

		self.pi_D = np.expand_dims(pi, axis=0).repeat(self.N, axis=0)
		self.pi_D[self.pi_D == 0.] = 1e-7
		self.pi_D[self.pi_D == 1.] = 1 - 1e-7
		self.logit_pi_D = np.log(self.pi_D / (1. - self.pi_D))

	def update_pi_S(self):
		""" Empirical Bayes update of the hyperparameter pi_S
		"""
		# pi is PxK
		pi = np.mean(self.p_S, axis=1)

		self.pi_S = np.expand_dims(pi, axis=1).repeat(self.K, axis=1)
		self.pi_S[self.pi_S == 0.] = 1e-7
		self.pi_S[self.pi_S == 1.] = 1 - 1e-7
		self.logit_pi_S = np.log(self.pi_S / (1. - self.pi_S))

	def update_parameters(self, it):
		# sample data point uniformly from the data set
		mb_idx = np.random.randint(self.N, size=self.minibatch_size)

		# update the local variables
		self.a = self.update_a(self.a, self.X, self.p_D, self.r, mb_idx=mb_idx)
		if self.zi:
			self.p_D = self.update_p_D(self.p_D, self.X, self.a, self.r, mb_idx=mb_idx)
		self.r = self.update_r(self.r, self.X, self.a, self.p_D, mb_idx=mb_idx)
		
		# update global variables
		step_size = (it+1. + self.delay)**(-self.forget_rate)
		self.update_b(mb_idx, step_size)
		if self.sparse:
			self.update_p_S(mb_idx, step_size)

		if self.empirical_bayes:
			# update hyperparameters
			self.update_alpha(mb_idx)
			self.update_beta()

			if self.zi:
				self.update_pi_D(mb_idx)
			if self.sparse:
				self.update_pi_S()