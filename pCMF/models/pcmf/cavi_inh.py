""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model.

The variational parameter updates were all derived from Equation (40) of Blei, D. et al 2016. 
Basically, each parameter of the variational approximation of some latent variable is set as 
the expected value of the natural parameter of that variable's complete conditional.
"""

import numpy as np
from scipy.special import digamma, factorial
from pCMF.models.pcmf.klqp import KLqp
from pCMF.misc.utils import psi_inverse

class CoordinateAscentVI(KLqp):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def update_a(self, a, X, p, r):
		N = X.shape[0]

		for i in range(N):
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for j in range(self.P):
					total1 = total1 + p[i, j] * X[i, j] * r[i, j, k]
					total2 = total2 + p[i, j] * self.b[0, j, k] / self.b[1, j, k]
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
					total1 = total1 + self.p[i, j] * self.X[i, j] * self.r[i, j, k]
					total2 = total2 + self.p[i, j] * self.a[0, i, k] / self.a[1, i, k]
				self.b[0, j, k] = self.beta[0, j, k] + total1
				self.b[1, j, k] = self.beta[1, j, k] + total2
		#total = 0.
		#for j in range(self.P):
		#	for k in range(self.K):
		#		for i in range(self.N):
		#			total = total + p[i, j] * X[i, j] * r[i, j, k]
		#for i in range(self.N):
		#	total = total + np.expand_dims(self.p[i, :], 1) * np.matmul(np.expand_dims(self.X[i, :], 1).T, self.r[i, :, :])
		#self.b[0] = self.beta[0] + total

		#self.b[1] = self.beta[1] + np.matmul(self.p.T, self.a[0]/self.a[1])

	def update_p(self, p, X, a, r):
		N = X.shape[0]

		#logit_p = self.logit_pi - np.matmul(self.a[0]/self.a[1], (self.b[0]/self.b[1].T))
		logit_p = np.zeros((N, self.P))
		for i in range(N):
			for j in range(self.P):
				logit_p[i, j] = self.logit_pi[i, j] - np.sum(a[0, i, :]/a[1, i, :] * self.b[0, j, :]/self.b[1, j, :])
		p = np.exp(logit_p) / (1. + np.exp(logit_p))
		p[X != 0] = 1. - 1e-7
		p[p == 1.] = 1 - 1e-7
		p[p == 0.] = 1e-7

		return p

	def update_r(self, r, X, a, p):
		N = X.shape[0]

		ar = digamma(a[0]) - np.log(a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		aux = np.zeros((self.K,))	
		for i in range(N):
			for j in range(self.P):
				aux = np.exp(ar[i, :] + br[j, :])
				#for k in range(self.K):
				#	aux[k] = np.exp(a[i, k] + b[j, k])
				r[i, j,:] = aux / np.sum(aux)

		return r

	def update_pi(self):
		""" Empirical Bayes update of the hyperparameter pi
		"""
		# pi is NxP
		pi = np.mean(self.p, axis=0)

		self.pi = np.expand_dims(pi, axis=0).repeat(self.N, axis=0)
		self.logit_pi = np.log(self.pi / (1. - self.pi))

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

	def update_parameters(self, *args):
		# update the local variables
		self.a = self.update_a(self.a, self.X, self.p, self.r)
		self.p = self.update_p(self.p, self.X, self.a, self.r)
		self.r = self.update_r(self.r, self.X, self.a, self.p)

		# update global variables
		self.update_b()

		if self.empirical_bayes:
			# update hyperparameters
			self.update_pi()
			self.update_alpha()
			self.update_beta()
