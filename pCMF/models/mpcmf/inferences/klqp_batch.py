""" This file contains an abstract class for inference of the parameters of the variational
distributions that approximate the posteriors of the Single Cell Matrix Factorization
model.
"""

import time
import numpy as np
from scipy.special import factorial
from sklearn.metrics import silhouette_score
from pCMF.misc.utils import log_likelihood_L_batches

from abc import ABC, abstractmethod

class KLqp(ABC):
	def __init__(self, X, alpha, beta, b_train=None, b_test=None, X_test=None, pi_D=None, pi_S=None, mu_lib=None, var_lib=None, empirical_bayes=False):
		self.X = X
		self.N = X.shape[0] # no of observations
		self.P = X.shape[1] # no of genes
		self.K = alpha.shape[1] # latent space dim

		self.scaling = mu_lib is not None and var_lib is not None
		self.zi = pi_D is not None
		self.sparse = pi_S is not None
		self.batches = b_train is not None

		# If counts are big, clip the log-likelihood to avoid inf values
		self.clip_ll = np.any(np.isinf(factorial(self.X)))

		# Empirical Bayes estimation of hyperparameters
		self.empirical_bayes = empirical_bayes

		## Hyperparameters		
		self.alpha = np.expand_dims(alpha, axis=1).repeat(self.N, axis=1) # 2xNxK
		self.beta = beta # 2xPxK
		
		# cell-specific scalings
		# cell-specific scalings
		self.nu = np.ones((2, self.N))
		if self.scaling:
			print('Considering cell-specific scalings.')
			self.nu[0] = mu_lib**2 / var_lib
			self.nu[1] = mu_lib / var_lib
			if nb:
				print('Considering NB structure.')
				# Force NB by making shape=rate on the scaling factor (see https://arxiv.org/abs/1801.01708)
				# This makes a Gamma with mean 1 and variance 1/alpha
				# Inverse dispersion parameter is 
				alpha = 1.
				self.nu[0] = alpha * np.ones((self.N,))
				self.nu[1] = alpha * np.ones((self.N,))

		# zero-Inflation
		if self.zi:
			print('Considering zero-inflated counts.')
			self.pi_D = np.expand_dims(pi_D, axis=0).repeat(self.N, axis=0) # NxP
			self.logit_pi_D = np.log(self.pi_D / (1. - self.pi_D))
		else:
			self.pi_D = np.ones((self.N, self.P)) # NxP
		
		# sparsity
		if self.sparse:
			print('Considering loading sparsity.')
			self.pi_S =  np.expand_dims(pi_S, axis=1).repeat(self.K, axis=1) # PxK
			self.logit_pi_S = np.log(self.pi_S / (1. - self.pi_S))
		else:
			self.pi_S =  np.ones((self.P, self.K)) # PxK

		# batch correction
		if self.batches:
			self.b_train = b_train
			self.n_batches = b_train.shape[1]
			print('Considering {} experimental batches.'.format(self.n_batches))

		## Variational parameters
		self.a = np.ones((2, self.N, self.K)) + np.random.rand(2, self.N, self.K) # parameters of q(U)
		#self.a[1, :, :] = 1.
		self.b = np.ones((2, self.P, self.K)) + np.random.rand(2, self.P, self.K) # parameters of q(V)
		self.r = np.random.dirichlet([1 / self.K] * self.K, size=(self.N, self.P,)) # parameters of q(Z)
		
		if self.batches:
			self.b = np.ones((2, self.P, self.K + self.n_batches)) + np.random.rand(2, self.P, self.K + self.n_batches) # parameters of q(V)
			#self.b[1, :, :] = 1.
			self.r = np.random.dirichlet([1 / (self.K + self.n_batches)] * (self.K  + self.n_batches), size=(self.N, self.P,)) # parameters of q(Z)

		# cell-specific scalings
		self.n = np.ones((2, self.N)) # parameters of q(L), which is Gamma-distributed
		if self.scaling:
			 self.n = self.n + np.random.rand(2, self.N)

		# zero-inflation
		self.p_D = np.ones((self.N, self.P)) # parameters of q(D)
		if self.zi:
			 self.p_D = self.p_D * 0.5

		# sparsity
		self.p_S = np.ones((self.P, self.K)) # parameters of q(S)
		if self.sparse:
			 self.p_S = self.p_S * 0.9
			 self.logit_p_S = np.log(self.p_S / (1. - self.p_S))

		# Log-likelihood per iteration and per time unit
		self.train_ll_it = []
		self.train_ll_time = []

		self.test_ll_it = []
		self.test_ll_time = []

		# Silhouette per iteration and per time unit
		self.silh_it = []
		self.silh_time = []

	def estimate_L(self, n=None):
		if n is None:
			n = self.n
		return n[0] / n[1]

	def estimate_U(self, a=None):
		if a is None:
			a = self.a
		return a[0] / a[1]

	def estimate_V(self, b=None):
		if b is None:
			b = self.b
		return b[0] / b[1]

	def estimate_D(self, p_D=None, thres=0.5):
		if p_D is None:
			p_D = self.p_D
		D = np.zeros((self.N, self.P))
		D[p_D > thres] = 1.
		return D

	def estimate_S(self, p_S=None, thres=0.5):
		if p_S is None:
			p_S = self.p_S
		S = np.zeros((self.P, self.K))
		S[p_S > thres] = 1.
		return S

	def get_estimates(self, thres=0.5):
		ests = {'L': None, 'U': None, 'V': None, 'D': None, 'S': None}
		ests['L'] = self.estimate_L()
		ests['U'] = self.estimate_U()
		ests['V'] = self.estimate_V()
		ests['D'] = self.estimate_D(thres=thres)
		ests['S'] = self.estimate_S(thres=thres)
		return ests

	def compute_elbo(self, X=None, b_idx=None, idx=None, n_iterations=5):
		if X is not None:
			N = X.shape[0]

			# local optimization
			# a = np.copy(self.alpha) + np.random.gamma(2., 1.)
			# a = np.abs(np.ones((2, N, self.K)) + np.random.rand(2, N, self.K)) # parameters of q(U) for test data
			a = np.ones((2, N, self.K))
			a[0, :, :] = self.alpha[0, 0, :] + np.random.gamma(2., 1.)
			a[1, :, :] = self.alpha[1, 0, :]
			r = np.ones((N, self.P, self.K)) * 1./self.K # parameters of q(Z) for test data
			p_D = np.ones((N, self.P)) # parameters of q(D) for test data
			n = np.ones((2, N))

			if self.zi:
				p_D = p_D * 0.5

			if self.scaling:
				n[0, :] = self.nu[0, 0]
				n[1, :] = self.nu[1, 0]

			# Posterior approximation over the local latent variables for each
			# of the new data points.
			for i in range(n_iterations):
				r = self.update_r(r, X, a, p_D, n) # parameter of Multinomial
				a = self.update_a(a, X, p_D, r, n) # parameters of Gamma
				if self.zi:
					p_D = self.update_p_D(p_D, X, a, r, n) # parameters of Bernoulli
				if self.scaling:
					n = self.update_n(n, X, a, p_D, r) # parameters of Gamma
		else:
			# Use training data
			X = self.X
			b_idx = self.b_idx
			N = X.shape[0]
			r = np.copy(self.r)
			a = np.copy(self.a)
			p_D = np.copy(self.p_D)
			n = np.copy(self.n)

			if idx is not None:
				X = self.X[idx, :]
				N = X.shape[0]
				r = np.copy(self.r[idx, :, :])
				a = np.copy(self.a[:, idx, :])
				p_D = np.copy(self.p_D[idx, :])
				n = np.copy(self.n[:, idx])

		# Compute ELBO = energy - entropy
		E_u = np.einsum('np,npk->npk', X, r) # NxPxK
		E_z = a[0] / a[1] # NxK
		E_w = self.b[0] / self.b[1] # PxK
		E_d = p_D # NxP
		E_l = n[0]/n[1] # Nx1

		E_log_z = digamma(a[0]) - np.log(a[1]) # NxK
		E_log_w = digamma(self.b[0]) - np.log(self.b[1]) # PxK
		E_log_d = E_d # NxP
		E_log_l = digamma(n[0]) - np.log(n[1]) # Nx1

		E_log_pz = self.alpha[0, 0] * np.log(self.alpha[1, 0]) - gammaln(self.alpha[0, 0]) + (self.alpha[0, 0] - 1)*E_log_z - self.alpha[1, 0]*E_z # NxK
		E_log_pw = self.beta[0] * np.log(self.beta[1]) - gammaln(self.beta[0]) + (self.beta[0] - 1)*E_log_w - self.beta[1]*E_w # PxK
		E_log_pd = E_d*np.log(self.pi_D[0, :] + 1e-7) + (1.-E_d)*np.log(1.-self.pi_D[0, :] + 1e-7) # NxP
		E_log_pl = self.nu[0, 0] * np.log(self.nu[1, 0]) - gammaln(self.nu[0, 0]) + (self.nu[0, 0] - 1)*E_log_l - self.nu[1, 0]*E_l # NxP

		E_log_qz = a[0] * np.log(a[1]) - gammaln(a[0]) + (a[0] - 1)*E_log_z - a[1]*E_z
		E_log_qw = self.b[0] * np.log(self.b[1]) - gammaln(self.b[0]) + (self.b[0] - 1)*E_log_w - self.b[1]*E_w
		E_log_qd = E_d*np.log(E_d + 1e-7) + (1.-E_d)*np.log(1.-E_d + 1e-7)
		E_log_ql = n[0] * np.log(n[1]) - gammaln(n[0]) + (n[0] - 1)*E_log_l - n[1]*E_l

		energy = 0
		if self.scaling:
			energy = energy + np.sum(np.einsum('np,npk,nk->n', E_d, E_u[:, :, :self.K], E_log_z) + np.einsum('np,npk,pk->n', E_d, E_u[:, :, :self.K], E_log_w[:, :self.K]) + np.einsum('np,npj,n->n', E_d, E_u, E_log_l)
				+ np.einsum('np,npb,nb->n', E_d, E_u[:, :, self.K:], b_idx) + np.einsum('np,npb,pb->n', E_d, E_u[:, :, self.K:], E_log_w[:, self.K:])
				 - np.einsum('np,nj,pj,n->n', E_d, np.concatenate(E_z, b_idx, axis=1), E_w, E_l))
			energy = energy + np.sum(E_log_pl)
		else:
			energy = energy + np.sum(np.einsum('np,npk,nk->n', E_d, E_u[:, :, :self.K], E_log_z) + np.einsum('np,npk,pk->n', E_d, E_u[:, :, :self.K], E_log_w[:, :self.K])
				+ np.einsum('np,npb,nb->n', E_d, E_u[:, :, self.K:], b_idx) + np.einsum('np,npb,pb->n', E_d, E_u[:, :, self.K:], E_log_w[:, self.K:])
				- np.einsum('np,nj,pj->n', E_d, np.concatenate(E_z, b_idx, axis=1), E_w))
		energy =  energy + np.sum(E_log_pz)
		energy = energy + np.sum(E_log_pw)
		if self.zi:
			energy = energy + np.sum(E_log_pd)

		entropy = 0
		entropy = entropy + np.sum(gammaln(X + 1.))
		entropy = entropy + np.sum(np.einsum('npk->n', E_u * np.log(r)))
		entropy = entropy + np.sum(E_log_qz)
		entropy = entropy + np.sum(E_log_qw)
		if self.zi:
			entropy = entropy + np.sum(E_log_qd)
		if self.scaling:
			entropy = entropy + np.sum(E_log_ql)
		
		elbo = energy - entropy
		elbo = elbo / N # lower bound on per-sample log-likelihood

		return elbo


	def predictive_ll(self, X_test, b_test, n_iterations=10, S=100):
		""" Computes the average posterior predictive likelihood of data not used for 
		training: p(X_test | X). It uses the posterior parameter estimates and 
		Monte Carlo sampling to compute the integrals using S samples.
		"""
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

	def generate_from_posterior(return_all=False):
		U = utils.sample_gamma(self.a[0], self.a[1])
		V = utils.sample_gamma(self.b[0], self.b[1])
		L = utils.sample_gamma(self.n[0], self.n[1])

		if self.sparse:
			est_S = self.estimate_S(self.p_S)
			V = V * est_S

		R = np.matmul(U.T, V)
		if self.scaling:
			R = R * L
		X = np.random.poisson(R)

		if self.zi:
			D = utils.sample_bernoulli(p=self.p_D)
			Y = np.where(D == 1, np.zeros((self.N, self.P)), X)
		else:
			Y = X

		if return_all:
			return Y, U, V, D
		return Y

	@abstractmethod
	def update_parameters(self):
		pass

	def run(self, n_iterations=10, calc_ll=False, calc_silh=False, max_time=60, sampling_rate=1., tol=0.0005, clusters=None):
		if calc_silh:
			if clusters is None:
				print("Can't compute silhouette score without cluster assignments.")
				calc_silh = False

		old_ll = -np.inf
		improvement = tol
		# init clock
		start = time.time()
		init = start
		for it in range(n_iterations):
			self.update_parameters(it)

			# Update log-likelihood and silhouette
			if calc_ll:
				# Subsample the data to evaluate the ll in
				idx = range(self.N)
				est_L = self.estimate_L(self.n[:, idx])
				est_U = self.estimate_U(self.a[:, idx, :])
				est_V = self.estimate_V(self.b)
				est_S = self.estimate_S(self.p_S)

				ll_curr = log_likelihood_L_batches(self.X[idx], est_U, est_V, self.p_D[idx], est_S, est_L, self.b_train, clip=self.clip_ll)
				self.ll_it.append(ll_curr)
			
			if calc_silh:
				est_U = self.estimate_U(self.a)
				silh_curr = silhouette_score(est_U, clusters)
				self.silh_it.append(silh_curr)

			end = time.time()
			it_time = end - start
			
			if it_time >= sampling_rate - 0.1*sampling_rate:
				if calc_ll:
					self.ll_time.append(ll_curr)
				if calc_silh:
					self.silh_time.append(silh_curr)
				start = end

			# Print status
			elapsed = end-init
			m, s = divmod(elapsed, 60)
			h, m = divmod(m, 60)
			if calc_ll:
				if it!=0:
					improvement = (ll_curr - old_ll) / abs(old_ll)
				print("Iteration {0}/{1}. Log-likelihood: {2:.7f}. Improvement: {3:.7f}. Elapsed: {4:.0f}h{5:.0f}m{6:.0f}s".format(it+1, 
					n_iterations, ll_curr, improvement, h, m, s), end="\r")

				if improvement < tol:
					print("\nConvergence criterion reached.")
					break
				old_ll = ll_curr
			else:
				print("Iteration {0}/{1}. Elapsed: {2:.0f}h{3:.0f}m{4:.0f}s".format(it+1, n_iterations, h, m, s), end="\r")

			# If maximum run time has passed, stop
			if elapsed >= max_time:
				break

		print('')
