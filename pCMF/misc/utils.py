import numpy as np
from scipy.special import factorial, psi, digamma, polygamma

def generate_U(N, K, C=2, alpha=1., eps=5.):
	U = sample_gamma(alpha, 1., size=(K, N))
	clusters = np.zeros((N,))
	for c in range(C):
		clusters[int(c*N/C):int((c+1)*N/C)] = clusters[int(c*N/C):int((c+1)*N/C)] + c
		size = U[int(c*K/C):int((c+1)*K/C), int(c*N/C):int((c+1)*N/C)].shape
		U[int(c*K/C):int((c+1)*K/C), int(c*N/C):int((c+1)*N/C)] = sample_gamma(alpha + eps, 1., size=size)
	return U, clusters

def generate_V(P, K, noisy_prop=0., M=2, beta=4., eps=4.):
	P_0 = int((1. - noisy_prop) * P)

	V = np.zeros((P, K))

	if noisy_prop > 0.:
		# noisy genes
		size = V[(P-P_0):, :].shape
		V[(P-P_0):, :] = sample_gamma(0.7, 1, size=size)

	# ungrouped genes
	V[:P_0, :] = sample_gamma(beta, 1, size=(P_0, K))

	# grouped genes
	for m in range(M):
		size = V[int(m*P_0/M):int((m+1)*P_0/M), int(m*K/M):int((m+1)*K/M)].shape
		V[int(m*P_0/M):int((m+1)*P_0/M), int(m*K/M):int((m+1)*K/M)] = sample_gamma(beta + eps, 1., size=size)
	return V

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

def generate_sparse_data(N, P, K, U=None, C=2, alpha=1., eps_U=5., V=None, M=2, beta=4., eps_V=4., noisy_prop=0., zero_prob=0.5, return_all=False):
	if U is None:
		U, clusters = generate_U(N, K, C, alpha, eps_U)
	else:
		K = U.shape[0]
		N = U.shape[1]

	assert K > M

	if V is None:
		V = generate_V(P, K, noisy_prop, M, beta, eps_V)

	R = np.matmul(U.T, V.T) # KxN X PxK
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

def log_likelihood(X, est_U, est_V, est_p_D, est_S):
	""" Computes the log-likelihood of the model from the inferred latent variables.
	"""
	N = X.shape[0]

	ll = np.zeros(X.shape)

	est_V = est_V * est_S
	param = np.dot(est_U, est_V.T)
	
	idx = (X != 0)
	factor = np.log(factorial(X[idx]))
	#factor = X[idx]
	ll[idx] = X[idx] * np.log(param[idx]) - param[idx] - factor
	
	idx = (X == 0)
	ll[idx] = np.log(1.-est_p_D[idx] + est_p_D[idx] * np.exp(-param[idx]))

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
