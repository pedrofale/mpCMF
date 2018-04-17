""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model.

Here we use Stochastic Variational Inference.
"""

import numpy as np
from scipy.special import digamma

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

	def predictive_ll(self):
		""" Computes the predictive log-likelihood of the model with current
		variational parameters.
		"""

		
		raise NotImplementedError("Not sure how to compute the predictive LL yet.")

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
		self.p[i, :] = np.exp(logit_p) / (1. + np.exp(logit_p))		
		
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

	def run_svi(self, n_iterations=10, minibatch_size=10, return_elbo=True, verbose=True):
		""" Run stochastic variational inference and return 
		variational parameters. Assess convergence via the ELBO. 
		"""
		if minibatch_size > self.N:
			print("Warning: minibatch size can't be larger than the number of samples.")
			print("Setting minibatch size to 1.")
			minibatch_size = 1

		if return_elbo:			
			ELBO = []
			if verbose:
				print("ELBO per iteration:")

		delay = 1.
		forget_rate = 0.9
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
		
	
			if return_elbo:
				# compute the ELBO
				elbo_curr = self.compute_elbo()
				ELBO.append(elbo_curr)
				if verbose:
					print("it. %d/%d: %f" % (it, n_iterations, elbo_curr))
			if verbose:
				print("Iteration {}/{}".format(it+1, n_iterations), end="\r")	
		if return_elbo: 
			return ELBO

