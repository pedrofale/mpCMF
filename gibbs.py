""" This file contains a class for inference of the posterior distribution of the Probabilistic 
Count Matrix Factorization model using Gibbs sampling.
"""

import numpy as np
from utils import sample_gamma, sample_bernoulli

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

		# Variables
		self.U = np.ones((self.N, self.K))
		self.V = np.ones((self.P, self.K))
		self.Z = np.ones((self.N, self.P, self.K))
		self.D = np.ones((self.N, self.P))

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


	def sample_fullcond_U(self):
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
			
	def sample_fullcond_V(self):
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

	def sample_fullcond_Z(self):
		# Update Z from its full conditional
		for i in range(self.N):
			for j in range(self.P):
				rho = self.U[i, :] * self.V[j, :]
				rho = rho/np.sum(rho)
				self.Z[i, j, :] = np.random.multinomial(n=self.X[i, j], 
										pvals=rho)
		
	def sample_fullcond_D(self):
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

	def gibbs_sample(self):	
		self.sample_fullcond_U()
		self.sample_fullcond_V()
		self.sample_fullcond_D()
		self.sample_fullcond_Z()

	def run_gibbs(self, n_iterations=100, init_prior=True, verbose=True, return_likelihood=True):
		if return_likelihood:
			likelihood = []

		U_samples = []
		V_samples = []
		D_samples = []
		Z_samples = []
			
		if init_prior:
			# Initialize variables with prior samples
			self.U = self.sample_prior_U()		
			self.V = self.sample_prior_V()
			self.D = self.sample_prior_D()
			self.Z = self.sample_prior_Z()
			
			if return_likelihood:
				likelihood.append(self.log_likelihood())
		
			U_samples.append(self.U)	
			V_samples.append(self.V)
			D_samples.append(self.D)
			Z_samples.append(self.Z)

		# Update variables by sampling iteratively from each full conditional
		for n in range(1, n_iterations):
			if verbose: 
				print("Iteration {}/{}".format(n+1, n_iterations), end="\r")
	
			self.sample_fullcond_U()
			self.sample_fullcond_V()
			self.sample_fullcond_D()
			self.sample_fullcond_Z()
			
			if return_likelihood:
				likelihood.append(self.log_likelihood())
						
			U_samples.append(self.U)	
			V_samples.append(self.V)
			D_samples.append(self.D)
			Z_samples.append(self.Z)
	
		if return_likelihood:
			return U_samples, V_samples, D_samples, Z_samples, likelihood

		return U_samples, V_samples, D_samples, Z_samples
	
	def log_likelihood(self):
		# likelihood = Poisson(D_ij * UV.T)
		param = self.D * (np.dot(self.U, self.V.T))
		ll = 0.
		for i in range(self.N):
			for j in range(self.P):
				ll = ll + self.X[i, j]*np.log(1. + param[i, j]) - param[i, j] - self.X[i, j]
		return ll
