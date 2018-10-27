""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Single Cell Matrix Factorization
model.

The variational parameter updates were all derived from Equation (40) of Blei, D. et al 2016. 
Basically, each parameter of the variational approximation of some latent variable is set as 
the expected value of the natural parameter of that variable's complete conditional

Here we use Stochastic Variational Inference.

"""

import numpy as np
from scipy.special import digamma, factorial
from pCMF.models.mpcmf.inferences.klqp import KLqp
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

	def update_n(self, n, X, a, p_D, r, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			idx = range(N)
		else:
			idx = mb_idx

		n[0, idx] = self.nu[0, 0] + np.einsum('ij,ijk->i', p_D[idx, :] * X[idx, :], r[idx, :, :])
		n[1, idx] = self.nu[1, 0] + np.einsum('ij,ik,jk->i', p_D[idx, :], (a[0]/a[1])[idx, :], self.p_S * self.b[0]/self.b[1])

		return n

	def update_a(self, a, X, p_D, r, n, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			idx = range(N)
		else:
			idx = mb_idx

		Lmean = (n[0]/n[1])[:, np.newaxis]
		a[0, idx, :] = self.alpha[0, 0, :] + np.einsum('ij,ijk->ik', p_D[idx,:] * X[idx,:], r[idx, :, :])
		a[1, idx, :] = self.alpha[1, 0, :] + Lmean[idx, :] * np.einsum('ij,jk->ik', p_D[idx, :], self.p_S * self.b[0]/self.b[1])

		return a

	def update_b(self, mb_idx, eta):
		S = mb_idx.size
		intermediate_b = np.ones((S, 2, self.P, self.K))

		Lmean = (self.n[0]/self.n[1])[:, np.newaxis]
		intermediate_b[:, 0, :, :] = self.beta[0, 0, :] + self.N * self.p_S * np.einsum('ijk,ij->ijk', self.r[mb_idx, :, :], self.p_D[mb_idx, :] * self.X[mb_idx, :])
		intermediate_b[:, 1, :, :] = self.beta[1, 0, :] + self.N * self.p_S * np.einsum('ij,ik->ijk', Lmean[mb_idx, :] * self.p_D[mb_idx, :], (self.a[0]/self.a[1])[mb_idx, :])

		# for s in range(S):
		# 	i = mb_idx[s]
		# 	for j in range(self.P):
		# 		for k in range(self.K):
		# 			intermediate_b[s, 0, j, k] = self.beta[0, j, k] + self.p_S[j, k] * self.N * self.p_D[i, j] * self.X[i, j] * self.r[i, j, k]
		# 			intermediate_b[s, 1, j, k] = self.beta[1, j, k] + self.p_S[j, k] * self.N *  Lmean[i, 0] * self.p_D[i, j] * self.a[0, i, k] / self.a[1, i, k]

		self.b = (1.-eta)*self.b + eta*np.mean(intermediate_b, axis=0)

	def update_p_D(self, p_D, X, a, r, n, mb_idx=None):
		N = X.shape[0]
		if mb_idx is None:
			idx = range(N)
		else:
			idx = mb_idx

		logit_p_D = np.zeros((N, self.P))

		Lmean = (n[0]/n[1])[:, np.newaxis]
		logit_p_D[idx, :] = self.logit_pi_D[idx, :] - np.einsum('jk,ik->ij', self.p_S * self.b[0]/self.b[1], (Lmean * a[0]/a[1])[idx, :])
		
		p_D = np.exp(logit_p_D) / (1. + np.exp(logit_p_D))
		p_D[X != 0] = 1. - 1e-30
		p_D[p_D == 1.] = 1 - 1e-30
		p_D[p_D == 0.] = 1e-30

		return p_D

	def update_r(self, r, X, a, p_D, n, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			idx = range(N)
		else:
			idx = mb_idx

		nr = digamma(n[0]) - np.log(n[1]) # N
		nr = nr[:, np.newaxis]
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
		r[idx, :, :] = np.einsum('ik,jk->ijk', np.exp(ar[idx, :]), np.exp(br)) + 1e-7
		r = r / (np.sum(r, axis=2)[:, :, np.newaxis]) # + 1 prevents value error
		return r

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
		# mb_idx = np.random.choice(self.N, size=self.minibatch_size, replace=False)

		self.r = self.update_r(self.r, self.X, self.a, self.p_D, self.n, mb_idx=mb_idx)

		assert np.all(self.r != 0)
		# update the local variables
		for i in range(1):
			self.a = self.update_a(self.a, self.X, self.p_D, self.r, self.n, mb_idx=mb_idx)
			if self.zi:
				self.p_D = self.update_p_D(self.p_D, self.X, self.a, self.r, self.n, mb_idx=mb_idx)
			if self.scaling:
				self.n = self.update_n(self.n, self.X, self.a, self.p_D, self.r, mb_idx=mb_idx)
		
		# update global variables
		step_size = (it+1. + self.delay)**(-self.forget_rate)
		self.update_b(mb_idx, step_size)
		if self.sparse:
			self.update_p_S(mb_idx, step_size)

		if self.empirical_bayes:
			# update hyperparameters
			self.update_alpha(mb_idx)
			# self.update_beta()

			# if self.scaling:
			# 	self.update_nu(mb_idx)
			if self.zi:
				self.update_pi_D(mb_idx)
			if self.sparse:
				self.update_pi_S()

		return mb_idx


	# def next_batch(X, batch_size):
	#    # Shuffle data
	#    shuffle_indices = np.random.permutation(np.arange(X.shape[0]))
	#    source = np.copy(X[shuffle_indices])

	#    for batch_i in range(0, X.shape[0])//batch_size):
	#       start_i = batch_i * batch_size
	#       source_batch = source[start_i:start_i + batch_size]

	#       yield np.array(source_batch)