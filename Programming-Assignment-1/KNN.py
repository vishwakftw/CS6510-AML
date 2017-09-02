import numpy as np

class KNN(object):
	"""
		Class to cover the implementation of K-Nearest Neighbours.
	"""
	def __init__(self, K, metrics):
		""" 
			Constructor for the KNN class. 
			Args:
				K		= The value of K i.e., the number of nearest neighbours to take into consideration
				metrics		= List of options from 'recall', 'precision' and 'accuracy'
							Example:
								metrics=['recall', 'accuracy']
									includes recall and accuracy
		"""
		super(KNN, self).__init__()
		
		self.K		= K
		self.metrics	= metrics
		
	def fit(X, Y, get_metrics=True):
		"""
			The function to fit the training data. This will be used later.
			Args:
				X		= The training inputs  : numpy.ndarray of shape (n_points, n_params)
				Y		= The training outputs : numpy.ndarray of shape (n_points, )
				get_metrics	= Get the metrics on the training dataset
		"""
		self.X	= X
		self.Y	= Y

		if get_metrics == True:
			predict	= []
			for i in range(0, self.X.shape[0]):
				distances 	= []
				for j in range(0, self.X.shape[0]):
					distances.append(np.linalg.norm(self.X[i] - self.X[j]))
				knns	= np.argpartition(distances, self.K)[:K].tolist()
				knn_lbl	= []
				for n in knns:
					knn_lbl.append(self.Y[n])
				prdctn	= 1 if np.sum(knn_lbl) > self.K/2 else 0
				predict.append(prdctn)
			
			metric_vals	= get_metric(self.metrics, self.Y, predict)
			for m in self.metrics:
				print('{0}: {1}'.format(m, metric_vals[m]))
	
	def predict(new_X, new_Y, get_metrics=True):
		"""
			The function to fit a batch of testing data.
			Args:
				new_X		= The testing inputs	: numpy.ndarray of shape (n_points, n_params)
				new_Y		= The testing outputs	: numpy.ndarray of shape (n_points, )
				get_metrics	= Get the metrics on the testing batch
		"""
		predict	= []
		for i in range(0, self.X.shape[0]):
			distances 	= []
			for j in range(0, self.X.shape[0]):
				distances.append(np.linalg.norm(self.X[i] - self.X[j]))
			knns	= np.argpartition(distances, self.K)[:K].tolist()
			knn_lbl	= []
			for n in knns:
				knn_lbl.append(self.Y[n])
			prdctn	= 1 if np.sum(knn_lbl) > self.K/2 else 0
			predict.append(prdctn)
			
		metric_vals	= get_metric(self.metrics, self.Y, predict)
		for m in self.metrics:
			print('{0}: {1}'.format(m, metric_vals[m]))
		
def get_confusion_matrix(original, predicted, positive=1):
	"""
		Function to get some confusion matrix
		Args:
			original	= The real outputs
			predicted	= The predicted outputs
			positive	= (Optional, default = 1) The value corresponding to positive
	"""
	n_pts		= len(original)
	assert n_pts == len(predicted), "Wrong dimensions"
	negative	= 0 if positive == 1 else 1

	TP = FN = TN = FP = 0
	for i in range(0, n_pts):
		if original[i] == negative:
			if predicted[i] == negative:
				TN = TN + 1
			elif predicted[i] == positive:
				FP = FP + 1
		elif original[i] == positive:
			if predicted[i] == negative:
				FN = FN + 1
			elif predicted[i] == positive:
				TP = TP + 1 
	
	return TP, FN, TN, FP

def get_metrics(names, original, predicted):
	"""
		Function to get metrics
		Args:
			names		= Names of the metrics	: list
			original	= The real outputs	: numpy.ndarray of shape (n_points, )
			predicted	= The predicted outputs	: numpy.ndarray of shape (n_points, )
	"""
	TP, FN, TN, FP	= get_confusion_matrix(original=original, predicted=predicted)
	ret_vals	= {}
	for n in names:
		if n == 'accuracy':
			ret_vals[n] = (TP + TN)/(TP + FN + TN + FP)
		if n == 'recall':
			ret_vals[n] = (TP)/(TP + FN)
		if n == 'precision':
			ret_vals[n] = (TP)/(TP + FP)
			
	return ret_vals
