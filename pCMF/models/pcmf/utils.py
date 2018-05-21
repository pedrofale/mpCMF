import numpy as np
from scipy.special import factorial

def log_likelihood(X, est_U, est_V, est_p):
	""" Computes the log-likelihood of the model from the inferred latent variables.
	"""
	N = X.shape[0]

	ll = np.zeros(X.shape)
	param = np.dot(est_U, est_V.T)
	
	idx = (X != 0)
	factor = np.log(factorial(X[idx]))
	#factor = X[idx]
	ll[idx] = X[idx] * np.log(param[idx]) - param[idx] - factor
	
	idx = (X == 0)
	ll[idx] = np.log(1.-est_p[idx] + est_p[idx] * np.exp(-param[idx]))

	ll = np.mean(ll)

	return ll

def generate_data(N, P, K, U=None, C=2, alpha=1., eps=5., shape=2., rate=2., zero_prob=0.5, return_all=False):
	if U is None:
		U, clusters = generate_U(N, K, C, alpha, eps)
	else:
		K = U.shape[0]
		N = U.shape[1]

	V = sample_gamma(shape, rate, size=(K, P))
	R = np.matmul(U.T, V)
	X = np.random.poisson(R)

	D = sample_bernoulli(p=zero_prob, size=(N, P))
	Y = np.where(D == 1, np.zeros((N, P)), X)

	if return_all:
		return Y, D, X, R, V, U, clusters
	else:
		return Y