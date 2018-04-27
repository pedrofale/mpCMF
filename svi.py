""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model.

Here we use Stochastic Variational Inference.
"""

import time
import math
import numpy as np
from scipy.special import digamma, factorial

class StochasticVI(object):
	def __init__(self, X, alpha, beta, pi):
		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.shape[1] # latent space dim
		
		# Hyperparameters		
		self.alpha = np.expand_dims(alpha, axis=1).repeat(self.N, axis=1) # 2xNxK
		self.beta = beta # 2xPxK
		self.pi =  np.expand_dims(pi, axis=0).repeat(self.N, axis=0) # NxP
		self.logit_pi = np.log(self.pi / (1. - self.pi))

		# Variational parameters
		self.a = np.ones((2, self.N, self.K)) + np.random.rand(2, self.N, self.K)# parameters of q(U)
		self.b = np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K) # parameters of q(V)
		self.r = np.ones((self.N, self.P, self.K)) * 0.5 # parameters of q(Z)
		self.p = np.ones((self.N, self.P)) * 0.5 # parameters of q(D)

	def compute_elbo(self):
		""" Computes the Evidence Lower BOund with the current variational parameters.
		J(q)_param = E[logp(var|-)] - E[logq(var;param)] + const
		"""
		# complete conditional term
		elbo = 0.

		# entropy term
		
		return elbo

	def log_likelihood(self):
		""" Computes the log-likelihood of the model with current
		parameter estimates.
		"""
		est_U = self.a[0] / self.a[1]
		est_V = self.b[0] / self.b[1]

		ll = np.zeros(self.X.shape)
		param = np.dot(est_U, est_V.T)
		idx = (self.X != 0)
		ll[idx] = np.log(self.p[idx]) + self.X[idx] * np.log(param[idx]) - param[idx] - np.log(factorial(self.X[idx]))
		idx = (self.X == 0)
		ll[idx] = np.log(1-self.p[idx] + self.p[idx] * np.exp(-param[idx]))
		ll = np.mean(ll)

		# ll = 0.
		# for i in range(self.N):
		# 	for j in range(self.P):
		# 		param = np.dot(est_U[i,:], est_V[j, :].T)
		# 		if self.X[i, j] != 0:
		# 			ll = ll + np.log(self.p[i, j]) + self.X[i, j] * np.log(param) - param - math.log(math.factorial(self.X[i, j]))
		# 		else:
		# 			ll = ll + np.log(1-self.p[i,j] + self.p[i,j] * np.exp(-param))
		
		# ll = ll #/ self.N # average log likelihood over the samples

		return ll

	def update_a(self, minibatch_indexes):
		""" Update the vector [a_1, a_2] for all (i,k) pairs.
		
		a_ik1 = alpha_k1 + sum_j(E[D_ij]*E[Z_ijk])
		a_ik2 = alpha_k2 + sum_j(E[D_ij]*E[V_jk])
		
		Requires:
		alpha	-- 1xK: the prior alphas vector
		p	-- NxP: E[D] vector
		X	-- NxP: the data
		r	-- NxPxK: E[Z]/X vector 
		b	-- PxK: parameters of q(V)
		"""
		for i in minibatch_indexes:	
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for j in range(self.P):
					total1 = total1 + self.p[i, j] * self.X[i, j] * self.r[i, j, k]
					total2 = total2 + self.p[i, j] * self.b[0, j, k] / self.b[1, j, k]
				self.a[0, i, k] = self.alpha[0, i, k] + total1
				self.a[1, i, k] = self.alpha[1, i, k] + total2


	def update_b(self, minibatch_indexes, eta):
		""" Update the vector [b_1, b_2] for all (j,k) pairs using the ith local variables.
		
		b_jk1 = beta_k1 + N*E[D_ij]*E[Z_ijk]
		b_jk2 = beta_k2 + N*E[D_ij]*E[U_ik]
		
		Requires:
		beta	-- the prior betas vector
		p	-- E[D] vector
		X	-- the data
		r	-- E[Z]/X vector
		a	-- parameters of q(U)
		"""
		S = minibatch_indexes.size
		intermediate_b = np.ones((S, 2, self.P, self.K))

		for s in range(S):
			i = minibatch_indexes[s]
			for j in range(self.P):
				for k in range(self.K):
					intermediate_b[s, 0, j, k] = self.beta[0, j, k] + self.N * self.p[i, j] * self.X[i, j] * self.r[i, j, k]
					intermediate_b[s, 1, j, k] = self.beta[1, j, k] + self.N * self.p[i, j] * self.a[0, i, k] / self.a[1, i, k]
		
		self.b = (1-eta)*self.b + eta*np.mean(intermediate_b, axis=0)
		
	def update_p(self, minibatch_indexes):
		""" Update the vector p for all j for given i.
		
		logit(p_ij) = logit(pi_j) - sum_k(E[U_ik]*E[V_jk])
		
		Requires:
		pi	-- prior dropout probabilities
		a	-- parameters of q(U)
		b	-- parameters of q(V)
		"""
		#logit_p = self.logit_pi - np.matmul(self.a[0]/self.a[1], (self.b[0]/self.b[1].T))
		logit_p = np.zeros((self.P,))
		for i in minibatch_indexes:
			for j in range(self.P):
				logit_p[j] = self.logit_pi[i, j] - np.sum(self.a[0, i, :]/self.a[1, i, :] * self.b[0, j, :]/self.b[1, j, :])
		# self.p[i, :] = np.exp(logit_p) / (1. + np.exp(logit_p))  ###### This is wrong, right?
		self.p[minibatch_indexes, :] = np.exp(logit_p) / (1. + np.exp(logit_p))		
		
		self.p[self.X != 0] = 1.

	def update_r(self, minibatch_indexes):
		""" Update the vector r for all (i,j,k).
		
		r_ijk \prop exp(E[logU_ik] + E[logV_j)
		
		Note that, for X distributed as Gamma(a, b), E[logX] = digamma(a) - log(b)

		Requires:
		a	-- parameters of q(U)
		b	-- parameters of q(V)
		""" 
		a = digamma(self.a[0]) - np.log(self.a[1]) # NxK
		b = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		aux = np.zeros((self.K,))
		for i in minibatch_indexes:	
			for j in range(self.P):
				aux = np.exp(a[i, :] + b[j, :])
				#for k in range(self.K):
				#	aux[k] = np.exp(a[i, k] + b[j, k])
				self.r[i,j,:] = aux / np.sum(aux)

	def run_svi(self, n_iterations=10, minibatch_size=10, delay=1., forget_rate=0.9, return_ll=True, sampling_rate=10, max_time=60, verbose=True):
		""" Run stochastic variational inference and return 
		variational parameters. Assess convergence via the ELBO. 
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
	
			# update the local variables corresponding to the sampled data point
			self.update_a(mb_idx)
			self.update_p(mb_idx)
			self.update_r(mb_idx)

			# update global variables, considering an hypothetical data set 
			# containing N replicates of sample n and a new step_size
			step_size = (it + delay)**(-forget_rate)
			self.update_b(mb_idx, step_size)
		
			if return_ll:
				# compute the LL
				ll_curr = self.log_likelihood()
				end = time.time()
				it_time = end - start
				if it_time >= sampling_rate - 0.1*sampling_rate:
					ll_time.append(ll_curr)
					start = end
				ll_it.append(ll_curr)
				if verbose:
					print("Iteration {0}/{1}. Log-likelihood: {2:.3f}. Elapsed: {3:.0f} seconds".format(it+1, n_iterations, ll_curr, end-init), end="\r")
				if (end - init) >= max_time:
					break
			elif verbose:
				print("Iteration {}/{}".format(it+1, n_iterations), end="\r")	
		if return_ll: 
			return ll_it, ll_time

