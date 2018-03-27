""" This file contains a class for inference of the parameters of the variational
distributions that approximate the posteriors of the Probabilistic Count Matrix Factorization
model. 
"""

import numpy as np
from scipy.special import digamma

class Inference(object):
	def __init__(self, X, alpha, beta, pi):
		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.size # latent space dim
		
		# Hyperparameters		
		self.alpha = alpha.repeat(self.N, axis=1) # 2xNxK
		self.beta = beta # 2xPxK
		self.pi = pi.repeat(self.N, axis=0) # NxP
		self.logit_pi = np.log(self.pi / (1. - self.pi))

		# Variational parameters
		self.a = np.ones((2, self.N, self.K)) # parameters of q(U)
		self.b = np.ones((2, self.P, self.K)) # parameters of q(V)
		self.r = np.ones((self.N, self.P, self.K)) # parameters of q(Z)
		self.p = np.ones((self.N, self.P)) # parameters of q(D)

	def compute_elbo(self):
		""" Computes the Evidence Lower BOund with the current variational parameters.
		"""
		# joint density term
		elbo = 0.		

		# entropy term
		
		return elbo

	def predictive_ll(self):
		""" Computes the predictive log-likelihood of the model with current
		variational parameters.
		"""
		raise NotImplementedError("Not sure how to compute the predictive LL yet.")

	def update_a(self):
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
		
		total = 0.
		for j in range(self.P):
			total = total + np.expand_dims(self.p[:, j], 1) * np.matmul(np.expand_dims(self.X[:, j], 1).T, self.r[:, j, :])]
		self.a[0] = self.alpha[0] + total

		self.a[1] = self.alpha[1] + np.matmul(self.p, self.b[0]/self.b[1])

	def update_b(self):
		""" Update the vector [b_1, b_2] for all (j,k) pairs.
		
		b_jk1 = beta_k1 + sum_i(E[D_ij]*E[Z_ijk])
		b_jk2 = beta_k2 + sum_i(E[D_ij]*E[U_ik])
		
		Requires:
		beta	-- the prior betas vector
		p	-- E[D] vector
		X	-- the data
		r	-- E[Z]/X vector
		a	-- parameters of q(U)
		"""

		total = 0.
		for i in range(self.N):
			total = total + np.expand_dims(self.p[i, :], 1) * np.matmul(np.expand_dims(self.X[i, :], 1).T, self.r[i, :, :])]
		self.b[0] = self.beta[0] + total

		self.b[1] = self.beta[1] + np.matmul(self.p, self.a[0]/self.a[1])

	def update_p(self):
		""" Update the vector p for all (i,j) pairs.
		
		logit(p_ij) = logit(pi_j) - sum_k(E[U_ik]*E[V_jk])
		
		Requires:
		pi	-- prior dropout probabilities
		a	-- parameters of q(U)
		b	-- parameters of q(V)
		"""
		logit_p = self.logit_pi - np.matmul(self.a[0]/self.a[1], self.b[0]/self.b[1])
		self.p = np.exp(logit_p) / (1. - np.exp(logit_p))

	def update_r(self):
		""" Update the vector r for all (i,j,k).
		
		r_ijk \prop exp(E[logU_ik] + E[logV_j)
		
		Note that, for X distributed as Gamma(a, b), E[logX] = digamma(a) - log(b)

		Requires:
		a	-- parameters of q(U)
		b	-- parameters of q(V)
		""" 
		a = digamma(self.a[0]) - np.log(self.a[1]) # NxK
		b = digamma(self.b[0]) - np.log(self.b[1]) # PxK
			
		for i in range(N):
			for j in range(P):
				total = 0.
				for k in range(K):
					aux = np.exp(a[i, k] + b[j, k])
					total = total + aux
					self.r[i,j,k] = aux / total 

	def run_cavi(self, n_iterations=10, return_elbo=True):
		""" Run coordinate ascent variational inference and return 
		variational parameters. Assess convergence via the ELBO. 
		"""
		ELBO = []
		print("ELBO per iteration:")
		for it in range(n_iterations):
			# update the local variables
			for i in range(self.N):
				self.update_a()
				self.update_p()
				self.update_r()

			# update global variables
			self.update_b()	
			
			if return_elbo:
				# compute the ELBO
				elbo_curr = self.compute_elbo()
				ELBO.append(elbo_curr)
				print("it. %d/%d: %f" % (it, n_iterations, elbo_curr))

		if return_elbo: 
			return ELBO

def sample_gamma(shape, rate, size=None):
	return np.random.gamma(shape, 1./rate, size=size)

def sample_bernoulli(p, size=None):
	return np.random.binomial(1, p, size=size)
