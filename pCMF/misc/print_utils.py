import operator

def print_model_lls(model_list, mode='Train', filename=None):
	""" Print ordered train or test log-likelihoods.
	"""
	assert mode in ['Train', 'Test']

	f = None
	if filename is not None:
		f = open(filename, 'w')

	names = []
	lls = []
	for model in model_list:
		names.append(model.name)
		if mode=='Train':
			lls.append(model.train_ll)
		elif mode=='Test':
			lls.append(model.test_ll)
			assert model.test_ll is not None

	scores = dict(zip(names, lls))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('{} data log-likelihood:'.format(mode), file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()

def print_model_silhouettes(model_list, filename=None):
	""" Print ordered silhouette scores.
	"""
	f = None
	if filename is not None:
		f = open(filename, 'w')

	names = []
	silhs = []
	for model in model_list:
		names.append(model.name)
		silhs.append(model.silhouette)

	scores = dict(zip(names, silhs))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('Silhouette scores:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()


def print_model_silhouettes(model_list, filename=None):
	""" Print ordered silhouette scores.
	"""
	f = None
	if filename is not None:
		f = open(filename, 'w')

	names = []
	silhs = []
	for model in model_list:
		names.append(model.name)
		silhs.append(model.silhouette)

	scores = dict(zip(names, silhs))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('Silhouette scores:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()

def print_model_dropid_acc(model_list, filename=None):
	f = None
	if filename is not None:
		f = open(filename, 'w')

	names = []
	dropid_accs = []
	for model in model_list:
		names.append(model.name)
		dropid_accs.append(model.dropid_acc)

	scores = dict(zip(names, dropid_accs))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

	print('Dropout identification accuracy:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()

def print_model_dropimp_err(model_list, filename=None):
	f = None
	if filename is not None:
		f = open(filename, 'w')

	names = []
	dropimp_errs= []
	for model in model_list:
		names.append(model.name)
		dropimp_errs.append(model.dropimp_err)

	scores = dict(zip(names, dropimp_errs))

	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=False) # lower is better

	print('Dropout imputation error:', file=f)
	print('\033[1m- {0}: {1:.6}\033[0m'.format(sorted_scores[0][0], sorted_scores[0][1]), file=f)
	for score_tp in sorted_scores[1:]:
		print('- {0}: {1:.6}'.format(score_tp[0], score_tp[1]), file=f)

	if f is not None:
		f.close()