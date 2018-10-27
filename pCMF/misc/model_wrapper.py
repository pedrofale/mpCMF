from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.base import BaseEstimator
import numpy as np
from pCMF.misc import utils

# class ModelWrapper(object):
# 	def __init__(self, Y_train, c_train, D_train=None, X_train=None, Y_test=None, c_test=None, name="New model"):
# 		self.Y_train = Y_train
# 		self.D_train = D_train
# 		self.dropout_idx = None
# 		if D_train is not None:
# 			self.dropout_idx = np.where(self.D_train == 0)
# 		self.X_train = X_train
# 		self.c_train = c_train

# 		self.Y_test = Y_test
# 		self.c_test = c_test

# 		self.name = name
# 		self.est_U = None
# 		self.est_R = None
# 		self.est_D = None
# 		self.proj_2d = None
# 		self.silhouette = None
# 		self.ari = None
# 		self.nmi = None
# 		self.dropid_acc = None
# 		self.dropimp_err = None

# 	def run(self, run_func, est_real_func=None, est_drop_func=None, do_silh=True, do_tsne=True, do_imp=False):
# 		self.est_U = run_func(self.Y_train)

# 		if do_silh:
# 			self.silhouette = silhouette_score(self.est_U, self.c_train)

# 			C = np.unique(self.c_train).size
# 			kmeans = KMeans(n_clusters=C)
# 			res = kmeans.fit_predict(self.est_U)

# 			self.ari = adjusted_rand_score(self.c_train, res)

# 			self.nmi = normalized_mutual_info_score(self.c_train, res)

# 		if self.est_U.shape[1] > 2:
# 			if do_tsne:
# 				self.proj_2d = TSNE(n_components=2).fit_transform(self.est_U)
# 		else:
# 			self.proj_2d = self.est_U

# 		if do_imp:
# 			if self.D_train is not None and est_drop_func is not None:
# 				self.est_D = est_drop_func()
# 				self.dropid_acc = accuracy_score(self.est_D.flatten(), self.D_train.flatten())
# 				if self.X_train is not None and est_real_func is not None:
# 					self.est_R = est_real_func()
# 					self.dropimp_err = utils.imputation_error(self.X_train, self.est_R, self.dropout_idx)

class ModelWrapper(object):
	def __init__(self, model_inst, Y_train, c_train, b_train=None, D_train=None, X_train=None, Y_test=None, c_test=None, b_test=None, log_data=False, name=None):
		assert issubclass(model_inst.__class__, BaseEstimator)

		# If it is a BaseEstimator, it has the methods .fit, .transform, .fit_transform and .score
		self.model_inst = model_inst

		self.log_data = log_data
		if log_data:
			print('Will assume the logarithm has been applied to the data.')

		self.Y_train = Y_train
		self.D_train = D_train
		self.dropout_idx = None
		if D_train is not None:
			self.dropout_idx = np.where(self.D_train == 0)
		self.X_train = X_train
		self.c_train = c_train
		self.b_train = b_train
		self.batches = b_train is not None

		self.Y_test = Y_test
		self.c_test = c_test
		self.b_test = b_test

		if name is None:
			name = self.model_inst.__class__.__name__
		self.name = name
		self.est_U = None
		self.est_R = None
		self.proj_2d = None
		self.silhouette = None
		self.asw = None
		self.ari = None
		self.nmi = None
		self.dropimp_err = 100000.
		self.dropid_acc = 0.
		self.train_ll = None
		self.test_ll = None
		self.batch_asw = None

	def run(self, max_iter=1, max_time=60, do_tsne=True, do_dll=True, do_holl=True, do_silh=True, do_imp=True, verbose=False):
		if verbose:
			print('Running {0}...'.format(self.name))

		try:
			self.est_U = self.model_inst.fit_transform(self.Y_train, batch_idx=self.b_train, max_time=max_time, max_iter=max_iter)
		except TypeError:
			#do stuff
			print('Some arguments were ignored by {}.'.format(self.model_inst.__class__.__name__))
			try:
				self.est_U = self.model_inst.fit_transform(self.Y_train, max_time=max_time, max_iter=max_iter)
			except TypeError:
				print('Running .fit_transform() without keyword arguments.')
				self.est_U = self.model_inst.fit_transform(self.Y_train)

		if do_silh:
			self.silhouette = silhouette_score(self.est_U, self.c_train)
			self.asw = self.silhouette
			if self.batches:
				if self.n_batches > 2:
					raise ValueError("Only 2 batches supported.")
				self.batch_asw = silhouette_score(self.est_U, self.b_train[:, 0]) # this only works for 2 batches!!!

			C = np.unique(self.c_train).size
			kmeans = KMeans(n_clusters=C)
			res = kmeans.fit_predict(self.est_U)

			self.ari = adjusted_rand_score(self.c_train, res)

			self.nmi = normalized_mutual_info_score(self.c_train, res)

		if self.est_U.shape[1] > 2:
			if do_tsne:
				self.do_tsne()
		else:
			self.proj_2d = self.est_U

		if do_dll:
			try:
				self.train_ll = self.model_inst.score(self.Y_train, batch_idx=self.b_train)
			except TypeError:
				self.train_ll = self.model_inst.score(self.Y_train)
			if self.log_data:
				self.train_ll = self.train_ll - np.mean(np.sum(self.Y_train, axis=-1))
		
		if do_holl:
			if self.Y_test is not None:
				try:
					self.test_ll = self.model_inst.score(self.Y_test, batch_idx=self.b_test)
				except TypeError:
					self.test_ll = self.model_inst.score(self.Y_test)
				if self.log_data:
					self.test_ll = self.test_ll - np.mean(np.sum(self.Y_test, axis=-1))

		if do_imp:
			if self.D_train is not None:
				self.est_R = self.model_inst.get_est_X()
				self.est_D = self.model_inst.get_est_D()

				self.dropid_acc = accuracy_score(self.est_D.flatten(), self.D_train.flatten())
				if self.X_train is not None:
					X_train = self.X_train
					if self.log_data:
						X_train = np.exp(self.X_train) - 1 # est_R is in count form
					self.dropimp_err = utils.imputation_error(X_train, self.est_R, self.dropout_idx)

		if verbose:
			print('Done.')

	def do_tsne(self):
		self.proj_2d = TSNE(n_components=2).fit_transform(self.est_U)