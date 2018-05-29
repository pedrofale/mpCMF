from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

class ModelWrapper(object):
	def __init__(self, X_train, c_train, D_train=None, name="New model"):
		self.X_train = X_train
		self.c_train = c_train

		self.name = name
		self.est_U = None
		self.proj_2d = None
		self.silhouette = None

	def run(self, run_func, do_silh=True, do_tsne=True):
		self.est_U = run_func(self.X_train)

		if do_silh:
			self.silhouette = silhouette_score(self.est_U, self.c_train)

		if do_tsne:
			self.proj_2d = TSNE(n_components=2).fit_transform(self.est_U)
