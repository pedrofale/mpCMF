import numpy as np

def generate_U(N, K, C=2, alpha=1., eps=5.):
	U = sample_gamma(alpha, 1., size=(K, N))
	clusters = np.zeros((N,))
	for c in range(C):
		clusters[int(c*N/C):int((c+1)*N/C)] = clusters[int(c*N/C):int((c+1)*N/C)] + c
		U[int(c*K/C):int((c+1)*K/C), int(c*N/C):int((c+1)*N/C)] = sample_gamma(alpha + eps, 1., size=(int(K/C), int(N/C)))
	return U, clusters

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

def sample_gamma(shape, rate, size=None):
	return np.random.gamma(shape, 1./rate, size=size)

def sample_bernoulli(p, size=None):
	return np.random.binomial(1., p, size=None)
