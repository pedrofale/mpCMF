""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Single Cell Matrix Factorization
model.

The variational parameter updates were all derived from Equation (40) of Blei, D. et al 2016. 
Basically, each parameter of the variational approximation of some latent variable is set as 
the expected value of the natural parameter of that variable's complete conditional.
"""

import numpy as np
from scipy.special import digamma, factorial
from pCMF.models.mpcmf.inferences.klqp import KLqp
from pCMF.misc.utils import psi_inverse

class CoordinateAscentVI(KLqp):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def update_n(self, n, X, a, p_D, r):
		n[0] = self.nu[0] + np.einsum('ij,ijk->i', p_D * X, r)
		n[1] = self.nu[1] + np.einsum('ij,ik,jk->i', p_D, a[0]/a[1], self.b[0]/self.b[1])

		return np.copy(n)

	def update_a(self, a, X, p_D, r, n):
		Lmean = (n[0]/n[1])[:, np.newaxis]
		a[0] = self.alpha[0] + np.einsum('ij,ijk->ik', p_D * X, r)
		a[1] = self.alpha[1] + Lmean * np.einsum('ij,jk->ik', p_D, self.b[0]/self.b[1])

		return np.copy(a)

	def update_b(self):
		Lmean = (self.n[0]/self.n[1])[:, np.newaxis]
		self.b[0] = self.beta[0] + np.einsum('ij,ijk->jk', self.p_D * self.X, self.r)
		self.b[1] = self.beta[1] + np.einsum('ij,ik->jk', Lmean * self.p_D, self.a[0]/self.a[1])

	def update_p_D(self, p_D, X, a, r, n):
		logit_p_D = np.zeros((X.shape[0], self.P))

		Lmean = (n[0]/n[1])[:, np.newaxis]

		logit_p_D = self.logit_pi_D[0, :] - np.einsum('jk,ik->ij', self.b[0]/self.b[1], Lmean * a[0]/a[1])
		# p_D[logit_p_D > 50] = 1.
		# p_D[logit_p_D < 50] = 0.
		# p_D[np.abs(logit_p_D) < 50] = np.exp(logit_p_D[np.abs(logit_p_D) < 50]) / (1. + np.exp(logit_p_D[np.abs(logit_p_D) < 50]))
		p_D = np.exp(logit_p_D) / (1. + np.exp(logit_p_D))
		p_D[X != 0] = 1. - 1e-30
		p_D[p_D == 1.] = 1 - 1e-30
		p_D[p_D == 0.] = 1e-30

		return np.copy(p_D)

	def update_r(self, r, X, a, p_D, n):
		N = X.shape[0]

		S = self.estimate_S(self.p_S)

		if self.scaling:
			nr = digamma(n[0]) - np.log(n[1]) # N
			nr = nr[:, np.newaxis]
		else:
			nr = np.ones([N, 1])

		ar = digamma(a[0]) - np.log(a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		# aux = np.zeros((self.K,))	
		# for i in range(N):
		# 	for j in range(self.P):
		# 		if np.all(S[j, :] == 0.):
		# 			r[i, j, :] = aux
		# 		else:
		# 			aux = S[j, :] * np.exp(ar[i, :] + br[j, :])
		# 			r[i, j, :] = aux / np.sum(aux)
		r = np.einsum('ik,jk->ijk', np.exp(ar), np.exp(br)) + 1e-7
		# we actually only need to infer r_np for X_np != 0

		r = r / (np.sum(r, axis=2)[:, :, np.newaxis]) # + 1 prevents value error

		assert np.all(r != 0)

		return np.copy(r)

	def update_p_S(self):
		ar = digamma(self.a[0]) - np.log(self.a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK

		Lmean = (self.n[0]/self.n[1])[:, np.newaxis]

		logit_p_S = np.zeros((self.P, self.K))
		for j in range(self.P):
			for k in range(self.K):
				logit_p_S[j, k] = self.logit_pi_S[j, k] + np.sum(-Lmean * self.p_D[:, j] * self.a[0, :, k]/self.a[1, :, k] * self.b[0, j, k]/self.b[1, j, k] +
					Lmean * self.p_D[:, j] * self.r[:, j, k] * self.X[:, j] * (ar[:, k] + br[j, k]))

		self.p_S[logit_p_S > 300] = 1.
		self.p_S[logit_p_S < 300] = np.exp(logit_p_S[logit_p_S < 300]) / (1. + np.exp(logit_p_S[logit_p_S < 300]))
		self.p_S[self.p_S == 0.] = 1e-7
		self.p_S[self.p_S == 1.] = 1 - 1e-7

		# x = np.einsum('ij,ik,jk->jk', -self.p_D, self.a[0]/self.a[1], self.b[0]/self.b[1])
		# y = np.einsum('ij,ijk,ik->jk', self.p_D * self.X, self.r, ar)
		# z = np.einsum('ij,ijk,jk->jk', self.p_D * self.X, self.r, br)
		# self.p_S = np.exp(self.logit_pi_S) * np.exp(x) * np.exp(y) * np.exp(z)
		
		# self.p_S[self.p_S == 0.] = 1e-7
		# self.p_S[self.p_S == 1.] = 1 - 1e-7

	def update_nu(self):
		""" Empirical Bayes update of the hyperparameter nu
		"""
		# self.nu[0] = np.log(self.nu[1]) + np.expand_dims(np.mean(digamma(self.n[0]) - np.log(self.n[1]), axis=0), axis=0).repeat(self.N, axis=0)
		# nu_1 = self.nu[0, 0]

		# nu_1 = psi_inverse(2., nu_1)

		# self.nu[0] = np.expand_dims(nu_1, axis=0).repeat(self.N, axis=0)
		self.nu[1] = self.nu[0] / np.mean(self.n[0] / self.n[1], axis=0)
		self.nu[0] = self.nu[1]

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
		self.r = self.update_r(self.r, self.X, self.a, self.p_D, self.n)

		# update the local variables
		self.a = self.update_a(self.a, self.X, self.p_D, self.r, self.n)

		if self.zi:
			self.p_D = self.update_p_D(self.p_D, self.X, self.a, self.r, self.n)
		if self.scaling:
			self.n = self.update_n(self.n, self.X, self.a, self.p_D, self.r)

		# update global variables
		self.update_b()

		if self.sparse:
			self.update_p_S()

		if self.empirical_bayes:
			# update hyperparameters
			# self.update_alpha()
			# self.update_beta()

			if self.scaling:
				self.update_nu()
			if self.zi:
				self.update_pi_D()
			if self.sparse:
				self.update_pi_S()
