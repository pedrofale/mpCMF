""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model.

Here we use Stochastic Variational Inference.

This version includes sparse factor loadings.

The variational parameter updates were all derived from Equation (40) of Blei, D. et al 2016. 
Basically, each parameter of the variational approximation of some latent variable is set as 
the expected value of the natural parameter of that variable's complete conditional.
"""

import time
import numpy as np
from scipy.special import digamma, factorial
from pCMF.misc.utils import log_likelihood, psi_inverse, log_likelihood_sparse

class StochasticVI(object):
	def __init__(self, X, alpha, beta, pi_D, pi_S):
		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.shape[1] # latent space dim
		
		# Hyperparameters		
		self.alpha = np.expand_dims(alpha, axis=1).repeat(self.N, axis=1) # 2xNxK
		self.beta = beta # 2xPxK
		self.pi_D =  np.expand_dims(pi_D, axis=0).repeat(self.N, axis=0) # NxP
		self.logit_pi_D = np.log(self.pi_D / (1. - self.pi_D))
		self.pi_S = np.expand_dims(pi_S, axis=1).repeat(self.K, axis=1) # PxK
		self.logit_pi_S = np.log(self.pi_S / (1. - self.pi_S))

		# Variational parameters
		self.a = np.ones((2, self.N, self.K)) + np.random.rand(2, self.N, self.K)# parameters of q(U)
		self.b = np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K) # parameters of q(V)
		self.r = np.ones((self.N, self.P, self.K)) * 0.5 # parameters of q(Z)
		self.p_D = np.ones((self.N, self.P)) * 0.5 # parameters of q(D)
		self.p_S = np.ones((self.P, self.K)) * 0.5 # parameters of q(S)
		self.logit_p_S = np.log(self.p_S / (1. - self.p_S))

	def estimate_U(self, a):
		return a[0] / a[1]

	def estimate_V(self, b):
		return b[0] / b[1]

	def estimate_S(self, p_S):
		S = np.zeros((self.P, self.K))
		S[p_S > 0.5] = 1.
		return S

	def predictive_ll(self, X_test, n_iterations=10, S=100):
		""" Computes the average posterior predictive likelihood of data not used for 
		training: p(X_test | X). It uses the posterior parameter estimates and 
		Monte Carlo sampling to compute the integrals using S samples.
		"""
		N_test = X_test.shape[0]

		a = np.ones((2, N_test, self.K)) + np.random.rand(2, N_test, self.K)# parameters of q(U) for test data
		r = np.ones((N_test, self.P, self.K)) * 0.5 # parameters of q(Z) for test data
		p = np.ones((N_test, self.P)) * 0.5 # parameters of q(D) for test data

		# Posterior approximation over the local latent variables for each
		# of the new data points.
		for i in range(n_iterations):
			r = self.update_r(r, X_test, a, p) # parameter of Multinomial
			a = self.update_a(a, X_test, p, r) # parameters of Gamma
			p = self.update_p_D(p, X_test, a, r) # parameters of Bernoulli
		
		# # Monte Carlo estimation of the posterior predictive: use S samples
		# U_samples = sample_gamma(a[0], a[1], size=S) # S by X_test.shape[0] by K
		# D_samples = sample_bernoulli(p, size=S) # S by X_test.shape[0] by P
		# V_samples = sample_gamma(self.b[0], self.b[1], size=S) # SxPxK

		est_U = self.estimate_U(a)
		est_V = self.estimate_V(self.b)
		est_S = self.estimate_S(self.p_S)

		pred_ll = log_likelihood_sparse(X_test, est_U, est_V, p, est_S) # S
		pred_ll = np.mean(pred_ll)
			
		return pred_ll

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
					total1 = total1 + p[i, j] * self.p_S[j, k] * X[i, j] * r[i, j, k]
					total2 = total2 + p[i, j] * self.p_S[j, k] * self.b[0, j, k] / self.b[1, j, k]
				a[0, i, k] = self.alpha[0, i, k] + total1
				a[1, i, k] = self.alpha[1, i, k] + total2

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

	def update_p_D(self, p, X, a, r, mb_idx=None):
		N = X.shape[0]	
		if mb_idx is None:
			iterator = range(N)
		else:
			iterator = mb_idx

		n = len(iterator)
		
		logit_p = np.zeros((N, self.P))
		for i in iterator:
			for j in range(self.P):
				logit_p[i, j] = self.logit_pi_D[i, j] - np.sum(self.p_S[j, :] * a[0, i, :]/a[1, i, :] * self.b[0, j, :]/self.b[1, j, :])
		
		if mb_idx is None:
			p[:, :] = np.exp(logit_p) / (1. + np.exp(logit_p))		
		else:
			p[mb_idx, :] = np.exp(logit_p[mb_idx, :]) / (1. + np.exp(logit_p[mb_idx, :]))

		p[X != 0] = 1. - 1e-7
		p[p == 1.] = 1 - 1e-7
		p[p == 0.] = 1e-7

		return p

	def update_p_S(self, minibatch_indexes, eta):
		S = minibatch_indexes.size
		intermediate_logit_p = np.ones((S, self.P, self.K))

		ar = digamma(self.a[0]) - np.log(self.a[1]) # NxK
		br = digamma(self.b[0]) - np.log(self.b[1]) # PxK

		logit_p = np.zeros((self.P, self.K))

		for s in range(S):
			i = minibatch_indexes[s]
			for j in range(self.P):
				for k in range(self.K):
					intermediate_logit_p[s, j, k] = self.logit_pi_S[j, k] + self.N *(-self.p_D[i, j] * self.a[0, i, k]/self.a[1, i, k] * self.b[0, j, k]/self.b[1, j, k] +
						self.p_D[i, j] * self.r[i, j, k] * self.X[i, j] * (ar[i, k] + br[j, k]))

		logit_p = (1.-eta)*self.logit_p_S + eta*np.mean(intermediate_logit_p, axis=0)		
		
		self.p_S[logit_p > 300] = 1.
		self.p_S[logit_p < 300] = np.exp(logit_p[logit_p < 300]) / (1. + np.exp(logit_p[logit_p < 300]))
		self.p_S[self.p_S == 0.] = 1e-7
		self.p_S[self.p_S == 1.] = 1 - 1e-7

		self.logit_p_S = np.log(self.p_S / (1. - self.p_S))

	def update_r(self, r, X, a, p, mb_idx=None):
		if mb_idx is None:
			N = X.shape[0]
			iterator = range(N)
		else:
			iterator = mb_idx

		S = np.zeros((self.P, self.K))
		S[self.p_S >= 0.5] = 1.

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

	def run_svi(self, X_test=None, empirical_bayes=False, n_iterations=10, minibatch_size=1, delay=1., forget_rate=0.9, return_ll=True, sampling_rate=10, max_time=60, verbose=True):
		""" Run stochastic variational inference and return 
		variational parameters.
		"""

		if minibatch_size > self.N:
			print("Warning: minibatch size can't be larger than the number of samples.")
			print("Setting minibatch size to 1.")
			minibatch_size = 1

		if return_ll:			
			ll_it = []
			ll_time = []

		# init clock
		start = time.time()
		init = start
		for it in range(n_iterations):
			# sample data point uniformly from the data set
			mb_idx = np.random.randint(self.N, size=minibatch_size)

			# update the local variables
			self.a = self.update_a(self.a, self.X, self.p_D, self.r, mb_idx=mb_idx)
			self.p_D = self.update_p_D(self.p_D, self.X, self.a, self.r, mb_idx=mb_idx)
			self.r = self.update_r(self.r, self.X, self.a, self.p_D, mb_idx=mb_idx)
			
			# update global variables
			step_size = (it+1. + delay)**(-forget_rate)
			self.update_b(mb_idx, step_size)
			self.update_p_S(mb_idx, step_size)

			if empirical_bayes:
				# update hyperparameters
				self.update_pi_D(mb_idx)
				self.update_pi_S()
				self.update_alpha(mb_idx)
				self.update_beta()
			
			if return_ll:
				# compute the LL
				if X_test is not None:
					ll_curr = self.predictive_ll(X_test)
				else:
					# subsample the data to evaluate the ll in
					idx = np.random.randint(self.N, size=np.min([100, self.N]))
					est_U = self.estimate_U(self.a[:, idx, :])
					est_V = self.estimate_V(self.b)
					est_S = self.estimate_S(self.p_S)

					ll_curr = log_likelihood_sparse(self.X[idx], est_U, est_V, self.p_D[idx], est_S)
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
