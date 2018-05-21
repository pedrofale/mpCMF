""" This file contains a class for inference of the posterior distribution of the Probabilistic 
Count Matrix Factorization model using Gibbs sampling.
"""

import time
import numpy as np
from scipy.special import factorial
from pCMF.misc.utils import sample_gamma, sample_bernoulli, log_likelihood

class GibbsSampling(object):
	def __init__(self, X, alpha, beta, pi):
		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.shape[1] # latent space dim

		# Hyperparameters
		self.alpha = np.expand_dims(alpha, axis=1).repeat(self.N, axis=1) # 2xNxK
		self.beta = beta # 2xPxK
		self.pi = np.expand_dims(pi, axis=0).repeat(self.N, axis=0) # NxP		

		# Initialize variables with prior samples
		self.U = self.sample_prior_U()		
		self.V = self.sample_prior_V()
		self.D = self.sample_prior_D()
		self.Z = self.sample_prior_Z()
		# self.U = np.ones((self.N, self.K))
		# self.V = np.ones((self.P, self.K))
		# self.Z = np.ones((self.N, self.P, self.K))
		# self.D = np.ones((self.N, self.P))

	def sample_prior_U(self):
		# NxK
		return sample_gamma(self.alpha[0], self.alpha[1])		
	
	def sample_prior_V(self):
		# PxK
		return sample_gamma(self.beta[0], self.beta[1])

	def sample_prior_Z(self):
		# NxPxK
		Z = np.zeros((self.N, self.P, self.K))
		for i in range(self.N):
			for j in range(self.P):
				for k in range(self.K):
					Z[i, j, k] = np.random.poisson(self.U[i, k] * self.V[j, k])
	
		return Z

	def sample_prior_D(self):
		# NxP
		# if X_ij is not zero, there was surely no dropout
		return sample_bernoulli(np.where(self.X != 0., np.ones((self.N, self.P)), self.pi))

	def fullcond_D_param(self):
		prob = np.zeros((self.N, self.P))
		for i in range(self.N):
			for j in range(self.P):
				if self.X[i, j] != 0:
					p = 1.
				else:
					e = np.exp(-np.dot(self.U[i, :], self.V[j, :].T))
					p = self.pi[i, j] * e / (1. - self.pi[i, j] + 
						self.pi[i, j] * e)
				prob[i, j] = p
		return prob

	def update_U(self):
		# Update U from its full conditional
		for i in range(self.N):
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for j in range(self.P):
					total1 = total1 + self.D[i, j] * self.Z[i, j, k]
					total2 = total2 + self.D[i, j] * self.V[j, k]
				self.U[i, k] = sample_gamma(self.alpha[0, i, k] + total1,
								self.alpha[1, i, k] + total2)
			
	def update_V(self):
		# Update V from its full conditional
		for j in range(self.P):
			for k in range(self.K):
				total1 = 0.
				total2 = 0.
				for i in range(self.N):
					total1 = total1 + self.D[i, j] * self.Z[i, j, k]
					total2 = total2 + self.D[i, j] * self.U[i, k]
				self.V[j, k] = sample_gamma(self.beta[0, j, k] + total1,
								self.beta[1, j, k] + total2)

	def update_Z(self):
		# Update Z from its full conditional
		for i in range(self.N):
			for j in range(self.P):
				rho = self.U[i, :] * self.V[j, :]
				rho = rho/np.sum(rho)
				self.Z[i, j, :] = np.random.multinomial(n=self.X[i, j], 
										pvals=rho)
		
	def update_D(self):
		# Update D from its full conditional
		for i in range(self.N):
			for j in range(self.P):
				if self.X[i, j] != 0:
					p = 1.
				else:
					e = np.exp(-np.dot(self.U[i, :], self.V[j, :].T))
					p = self.pi[i, j] * e / (1. - self.pi[i, j] + 
						self.pi[i, j] * e)
				self.D[i, j] = sample_bernoulli(p)

	def update_pi(self, M=100):
		""" Empirical Bayes update of the hyperparameter pi.

		D_samples: MxNxP
		"""
		# pi is NxP
		D_samples = np.zeros((M, self.N, self.P))
		for m in range(M):
			self.gibbs_sample()
			D_samples[m] = self.D

		pi = np.mean(D_samples, axis=(0,1))

		self.pi = np.expand_dims(pi, axis=0).repeat(self.N, axis=0)
		self.logit_pi = np.log(self.pi / (1. - self.pi))

	def gibbs_sample(self):	
		self.update_U()
		self.update_V()
		self.update_D()
		self.update_Z()
	
	def estimate_U(self, U_samples):
		return np.mean(U_samples, axis=0)

	def estimate_V(self, V_samples):
		return np.mean(V_samples, axis=0)

	def estimate_p(self, D_samples):
		return np.mean(D_samples, axis=0)

	def run_gibbs(self, X_test=None, empirical_bayes=False, n_iterations=10, mc_samples=100, return_ll=True, sampling_rate=10, max_time=60, verbose=True):
			""" Run Gibbs sampling and return posterior samples. 

			Evaluate the log-likelihood every <sampling_rate> seconds.

			If empirical_bayes, use mc_samples to compute the hyperparameter update
			"""
			U_samples = np.zeros((n_iterations, self.N, self.K))
			V_samples = np.zeros((n_iterations, self.P, self.K))
			D_samples = np.zeros((n_iterations, self.N, self.P))
			Z_samples = np.zeros((n_iterations, self.N, self.P, self.K))

			if return_ll:			
				ll_it = []
				ll_time = []

			# init clock
			start = time.time()
			init = start
			for it in range(n_iterations):				
				if empirical_bayes:
					self.update_pi(mc_samples)

				self.gibbs_sample()

				# add to list
				U_samples[it] = self.U	
				V_samples[it] = self.V
				D_samples[it] = self.D
				Z_samples[it] = self.Z
				
				if return_ll:
					# compute the LL
					# subsample the data to evaluate the ll in
					idx = np.random.randint(self.N, size=np.min([100, self.N]))

					est_p = self.fullcond_D_param()[idx, :]
					ll_curr = log_likelihood(self.X[idx], self.U[idx, :], self.V, est_p)
					
					end = time.time()
					it_time = end - start
					if it_time >= sampling_rate - 0.1*sampling_rate:
						ll_time.append(ll_curr)
						start = end
					ll_it.append(ll_curr)
					if verbose:
						print("Iteration {0}/{1}. Log-likelihood: {2:.3f}. Elapsed: {3:.0f} seconds".format(it, n_iterations, ll_curr, end-init), end="\r")
					if (end - init) >= max_time:
						break
				elif verbose:
					print("Iteration {}/{}".format(it+1, n_iterations), end="\r")	
			if verbose:
				print('')
			if return_ll: 
				return ll_it, ll_time
