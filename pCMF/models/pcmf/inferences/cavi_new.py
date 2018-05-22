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

class CoordinateAscentVI(KLqp):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def update_a(self, a, X, p_D, r):
		N = X.shape[0]

		for i in range(N):
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for j in range(self.P):
					total1 = total1 + p_D[i, j] * self.p_S[j, k] * X[i, j] * r[i, j, k]
					total2 = total2 + p_D[i, j] * self.p_S[j, k] * self.b[0, j, k] / self.b[1, j, k]
				a[0, i, k] = self.alpha[0, i, k] + total1
				a[1, i, k] = self.alpha[1, i, k] + total2

		return a

		#for j in range(self.P):
		#	total = total + np.expand_dims(self.p[:, j], 1) * np.matmul(np.expand_dims(self.X[:, j], 1).T, self.r[:, j, :])
		#self.a[0] = self.alpha[0] + total

		#self.a[1] = self.alpha[1] + np.matmul(self.p, self.b[0]/self.b[1])

	def update_b(self):
		for j in range(self.P):
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for i in range(self.N):
					total1 = total1 + self.p_D[i, j] * self.X[i, j] * self.r[i, j, k]
					total2 = total2 + self.p_D[i, j] * self.a[0, i, k] / self.a[1, i, k]
				self.b[0, j, k] = self.beta[0, j, k] + self.p_S[j, k] * total1
				self.b[1, j, k] = self.beta[1, j, k] + self.p_S[j, k] * total2
		#total = 0.
		#for j in range(self.P):
		#	for k in range(self.K):
		#		for i in range(self.N):
		#			total = total + p[i, j] * X[i, j] * r[i, j, k]
		#for i in range(self.N):
		#	total = total + np.expand_dims(self.p[i, :], 1) * np.matmul(np.expand_dims(self.X[i, :], 1).T, self.r[i, :, :])
		#self.b[0] = self.beta[0] + total

		#self.b[1] = self.beta[1] + np.matmul(self.p.T, self.a[0]/self.a[1])

	def update_p_D(self, p_D, X, a, r):
		N = X.shape[0]

		#logit_p = self.logit_pi - np.matmul(self.a[0]/self.a[1], (self.b[0]/self.b[1].T))
		logit_p_D = np.zeros((N, self.P))
		for i in range(N):
			for j in range(self.P):
				logit_p_D[i, j] = self.logit_pi_D[i, j] - np.sum(self.p_S[j, :] * a[0, i, :]/a[1, i, :] * self.b[0, j, :]/self.b[1, j, :])
		p_D = np.exp(logit_p_D) / (1. + np.exp(logit_p_D))
		p_D[X != 0] = 1. - 1e-7
		p_D[p_D == 1.] = 1 - 1e-7
		p_D[p_D == 0.] = 1e-7

		return p_D

	def update_r(self, r, X, a, p_D):
		N = X.shape[0]

		S = self.estimate_S(self.p_S)

		ar = digamma(a[0]) - np.log(a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		aux = np.zeros((self.K,))	
		for i in range(N):
			for j in range(self.P):
				if np.all(S[j, :] == 0.):
					r[i, j, :] = 0.
				else:
					aux = S[j, :] * np.exp(ar[i, :] + br[j, :])
					r[i, j, :] = aux / np.sum(aux)

		return r

	def update_p_S(self):
		ar = digamma(self.a[0]) - np.log(self.a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK

		logit_p_S = np.zeros((self.P, self.K))
		for j in range(self.P):
			for k in range(self.K):
				logit_p_S[j, k] = self.logit_pi_S[j, k] + np.sum(-self.p_D[:, j] * self.a[0, :, k]/self.a[1, :, k] * self.b[0, j, k]/self.b[1, j, k] +
					self.p_D[:, j] * self.r[:, j, k] * self.X[:, j] * (ar[:, k] + br[j, k]))

		self.p_S[logit_p_S > 300] = 1.
		self.p_S[logit_p_S < 300] = np.exp(logit_p[logit_p_S < 300]) / (1. + np.exp(logit_p_S[logit_p_S < 300]))
		self.p_S[self.p_S == 0.] = 1e-7
		self.p_S[self.p_S == 1.] = 1 - 1e-7

	def update_alpha(self):
		""" Empirical Bayes update of the hyperparameter alpha
		"""
		self.alpha[0] = np.log(self.alpha[1]) + np.expand_dims(np.mean(digamma(self.a[0]) - np.log(self.a[1]), axis=0), axis=0).repeat(self.N, axis=0)
		alpha_1 = self.alpha[0, 0, :]
		
		for k in range(self.K):
			alpha_1[k] = psi_inverse(2., self.alpha[0, 0, k])

		self.alpha[0] = np.expand_dims(alpha_1, axis=0).repeat(self.N, axis=0)
		self.alpha[1] = self.alpha[0] / np.mean(self.a[0] / self.a[1], axis=0)

	def update_beta(self):
		""" Empirical Bayes update of the hyperparameter beta
		"""
		self.beta[0] = np.log(self.beta[1]) + np.expand_dims(np.mean(digamma(self.b[0]) - np.log(self.b[1]), axis=0), axis=0).repeat(self.P, axis=0)
		beta_1 = self.beta[0, 0, :]

		for k in range(self.K):
			beta_1[k] = psi_inverse(2., self.beta[0, 0, k])

		self.beta[0] = np.expand_dims(beta_1, axis=0).repeat(self.P, axis=0)
		self.beta[1] = self.beta[0] / np.mean(self.b[0] / self.b[1], axis=0)

	def update_pi_D(self):
		""" Empirical Bayes update of the hyperparameter pi
		"""
		# pi is NxP
		pi = np.mean(self.p_D, axis=0)

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

	def update_parameters(self, *args):
		# update the local variables
		self.a = self.update_a(self.a, self.X, self.p_D, self.r)
		if self.zi:
			self.p_D = self.update_p_D(self.p_D, self.X, self.a, self.r)
		self.r = self.update_r(self.r, self.X, self.a, self.p_D)

		# update global variables
		self.update_b()
		if self.sparse:
			self.update_p_S()

		if self.empirical_bayes:
			# update hyperparameters
			self.update_alpha()
			self.update_beta()

			if self.zi:
				self.update_pi_D()
			if self.sparse:
				self.update_pi_S()