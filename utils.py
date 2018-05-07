import numpy as np
from scipy.special import factorial, psi, digamma, polygamma

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
	return np.random.binomial(1., p, size=size)

def log_likelihood(X, est_U, est_V, est_p):
	""" Computes the log-likelihood of the model from the inferred latent variables.
	"""
	N = X.shape[0]

	ll = np.zeros(X.shape)
	param = np.dot(est_U, est_V.T)
	
	idx = (X != 0)
	ll[idx] = np.log(est_p[idx]) + X[idx] * np.log(param[idx]) - param[idx] - np.log(factorial(X[idx]))
	
	idx = (X == 0)
	ll[idx] = np.log(1.-est_p[idx] + est_p[idx] * np.exp(-param[idx]))
	ll = np.mean(ll)

	return ll

def psi_inverse(initial_x, y, num_iter=5):
    """
    Computes the inverse digamma function using Newton's method
    See Appendix c of Minka, T. P. (2003). Estimating a Dirichlet distribution.
    Annals of Physics, 2000(8), 1-13. http://doi.org/10.1007/s00256-007-0299-1 for details.
    """

    # initialisation
    if y >= -2.22:
        x_old = np.exp(y)+0.5
    else:
        gamma_val = -psi(1)
        x_old = -(1/(y+gamma_val))

    # do Newton update here
    for i in range(num_iter):
        numerator = psi(x_old) - y
        denumerator = polygamma(1, x_old)
        x_new = x_old - (numerator/denumerator)
        x_old = x_new

    return x_new
