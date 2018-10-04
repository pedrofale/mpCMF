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
from pCMF.misc.utils import psi_inverse, log_likelihood_L_batches

class CoordinateAscentVI(KLqp):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def predictive_ll(self, n_iterations=10, S=100):
		""" Computes the average posterior predictive likelihood of data not used for 
		training: p(X_test | X). It uses the posterior parameter estimates and 
		Monte Carlo sampling to compute the integrals using S samples.
		"""
		X_test = self.X_test
		b_test = self.b_test

		N_test = X_test.shape[0]

		if self.batches:
			assert b_test is not None

		a = np.ones((2, N_test, self.K)) + np.random.rand(2, N_test, self.K)# parameters of q(U) for test data
		r = np.ones((N_test, self.P, self.K + self.n_batches)) * 0.5 # parameters of q(Z) for test data
		p_D = np.ones((N_test, self.P)) # parameters of q(D) for test data
		n = np.ones((2, N_test,))

		if self.zi:
			p_D = p_D * 0.5

		# Posterior approximation over the local latent variables for each
		# of the new data points.
		for i in range(n_iterations):
			r = self.update_r(r, X_test, a, p_D, n, b_test) # parameter of Multinomial
			a = self.update_a(a, X_test, p_D, r, n) # parameters of Gamma
			if self.zi:
				p_D = self.update_p_D(p_D, X_test, a, r, n, b_test) # parameters of Bernoulli
			if self.scaling:
				n = self.update_n(n, X_test, a, p_D, r, b_test) # parameters of Gamma
		
		# # Monte Carlo estimation of the posterior predictive: use S samples
		# U_samples = sample_gamma(a[0], a[1], size=S) # S by X_test.shape[0] by K
		# D_samples = sample_bernoulli(p, size=S) # S by X_test.shape[0] by P
		# V_samples = sample_gamma(self.b[0], self.b[1], size=S) # SxPxK

		est_U = self.estimate_U(a)
		est_V = self.estimate_V(self.b)
		est_S = self.estimate_S(self.p_S)
		est_L = self.estimate_L(n)

		pred_ll = log_likelihood_L_batches(X_test, est_U, est_V, p_D, est_S, est_L, b_test, clip=self.clip_ll) # S
		pred_ll = np.mean(pred_ll)
			
		return pred_ll

	def update_n(self, n, X, a, p_D, r, b_idx):
		# n[0, :] = self.nu[0, 0] + np.einsum('ij,ijk->i', p_D * X, r)
		n[0, :] = self.nu[0, 0] + np.einsum('ij,ijk->i', p_D * X, r)
		n[1, :] = self.nu[1, 0] + np.einsum('ij,ik,jk->i', p_D, np.concatenate((a[0]/a[1], b_idx), axis=1), self.b[0]/self.b[1])

		return n

	def update_a(self, a, X, p_D, r, n):
		Lmean = (n[0]/n[1])[:, np.newaxis]
		a[0] = self.alpha[0] + np.einsum('ij,ijk->ik', p_D * X, r[:, :, :self.K])
		a[1] = self.alpha[1] + np.einsum('ij,jk->ik', Lmean * p_D, (self.b[0]/self.b[1])[:, :self.K])

		return a

	def update_b(self):
		Lmean = (self.n[0]/self.n[1])[:, np.newaxis]
		self.b[0] = self.beta[0] + np.einsum('ijk,ij->jk', self.r, self.p_D * self.X)
		self.b[1] = self.beta[1] + np.einsum('ij,ik->jk', Lmean * self.p_D, np.concatenate((self.a[0]/self.a[1], self.b_train), axis=1))

	def update_p_D(self, p_D, X, a, r, n, b_idx):
		logit_p_D = np.zeros((X.shape[0], self.P))

		Lmean = (n[0]/n[1])[:, np.newaxis]

		logit_p_D = self.logit_pi_D[0, :] - np.einsum('jk,ik->ij', self.b[0]/self.b[1], Lmean * np.concatenate((a[0]/a[1], b_idx), axis=1))
		
		p_D = np.exp(logit_p_D) / (1. + np.exp(logit_p_D))
		p_D[X != 0] = 1. - 1e-7
		p_D[p_D == 1.] = 1 - 1e-7
		p_D[p_D == 0.] = 1e-7

		return p_D

	def update_r(self, r, X, a, p_D, n, b_idx):
		N = X.shape[0]

		# nr = digamma(n[0]) - np.log(n[1]) # N
		# nr = nr[:, np.newaxis]
		ar = digamma(a[0]) - np.log(a[1]) # NxK
		#ar = np.concatenate((ar, np.log(b_idx + 1e-7)), axis=1)
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		# aux = np.zeros((self.K,))	
		# for i in range(N):
		# 	for j in range(self.P):
		# 		if np.all(S[j, :] == 0.):
		# 			r[i, j, :] = aux
		# 		else:
		# 			aux = S[j, :] * np.exp(ar[i, :] + br[j, :])
		# 			r[i, j, :] = aux / np.sum(aux)
		r = np.einsum('ik,jk->ijk', np.concatenate((np.exp(ar), b_idx), axis=1), np.exp(br))

		r = r / (np.sum(r, axis=2)[:, :, np.newaxis]) # + 1 prevents value error
		return r

	def update_nu(self):
		""" Empirical Bayes update of the hyperparameter nu
		"""
		self.nu[0] = np.log(self.nu[1]) + np.expand_dims(np.mean(digamma(self.n[0]) - np.log(self.n[1]), axis=0), axis=0).repeat(self.N, axis=0)
		nu_1 = self.nu[0, 0]

		nu_1 = psi_inverse(2., nu_1)

		self.nu[0] = np.expand_dims(nu_1, axis=0).repeat(self.N, axis=0)
		self.nu[1] = self.nu[0] / np.mean(self.n[0] / self.n[1], axis=0)

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

	def update_parameters(self, *args):
		self.r = self.update_r(self.r, self.X, self.a, self.p_D, self.n, self.b_train)

		# update the local variables
		self.a = self.update_a(self.a, self.X, self.p_D, self.r, self.n)
		if self.zi:
			self.p_D = self.update_p_D(self.p_D, self.X, self.a, self.r, self.n, self.b_train)
		if self.scaling:
			self.n = self.update_n(self.n, self.X, self.a, self.p_D, self.r, self.b_train)

		# update global variables
		self.update_b()
		if self.sparse:
			self.update_p_S()

		if self.empirical_bayes:
			# update hyperparameters
			self.update_alpha()
			# self.update_beta()

			# if self.scaling:
			# 	self.update_nu()
			if self.zi:
				self.update_pi_D()
			if self.sparse:
				self.update_pi_S()