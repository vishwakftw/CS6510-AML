import NB
import sys
import numpy as np
from dataloader import get_dataset
from sklearn.naive_bayes import GaussianNB

def confusion_matrix(actual, predicted, positive=1):
	"""
		Function to return a confusion matrix
		Args:
			actual		= The actual outputs	: numpy.ndarray of shape (n_points, )
			predicted	= The predicted outputs	: numpy.ndarray of shape (n_points, )
			positive	= (Optional, default = 1) The value out of 0 and 1 to be considered as positive
		Returns:
			Dictionary with True Positive, True Negative, False Positive and False Negative readings
			Format:
				{'TP': <value>, 'TN': <value>, 'FP': <value>, 'FN': <value>}
	"""
	assert len(actual) == len(predicted), "Incorrect dimensions to compare"
	TP = TN = FP = FN = 0
	negative	= 0 if positive == 1 else 1

	for i in range(0, len(actual)):
		if actual[i] == positive:
			if predicted[i] == positive:
				TP	+= 1
			elif predicted[i] == negative:
				FN	+= 1
		elif actual[i] == negative:
			if predicted[i]	== positive:
				FP	+= 1
			elif predicted[i] == negative:
				TN	+= 1
	conf_dict	= {}
	conf_dict['TP'] = TP
	conf_dict['TN']	= TN
	conf_dict['FP']	= FP
	conf_dict['FN']	= FN

	return conf_dict

def get_metrics(metric_list, conf_mat):
	"""
		Function to calculate some well known metrics
		Args:
			metric_list	= List of metrics required. Current options: accuracy, recall, precision, f1score
			conf_mat	= confusion matrix dictionary generated from the confusion_matrix function
		Returns:
			dictionary with the metric values	
	"""
	ret_metrics	= {}

	if 'accuracy' in metric_list:
		ret_metrics['accuracy']	= (conf_mat['TP'] + conf_mat['TN'])/sum(conf_mat.values())

	elif 'recall' in metric_list:
		ret_metrics['recall']	= conf_mat['TP']/(conf_mat['TP'] + conf_mat['FN'])
		
	elif 'precision' in metric_list:
		ret_metrics['precision']= conf_mat['TP']/(conf_mat['TP'] + conf_mat['FP'])
		
	elif 'f1score' in metric_list:
		ret_metrics['f1score']	= (2*conf_mat['TP'])/(2*conf_mat['TP'] + conf_mat['FN'] + conf_mat['FP'])

	return ret_metrics

if len(sys.argv[1:]) != 1:
	print("Give the root folder as command line argument")
	sys.exit(1)
	
# Get the dataset
tr, te	= get_dataset(root=sys.argv[1])
tr_x, tr_y	= tr[ : , 0 : tr.shape[1]-1], tr[ : , tr.shape[1]-1]
te_x, te_y	= te[ : , 0 : te.shape[1]-1], te[ : , te.shape[1]-1]

my_NB	= NB.NB()
my_NB.fit(X=tr_x, Y=tr_y)
my_train_prediction, _	= my_NB.predict(X=tr_x)
my_test_prediction, _	= my_NB.predict(X=te_x)

the_NB	= GaussianNB()
the_NB.fit(X=tr_x, y=tr_y)
the_train_prediction, _	= the_NB.predict(X=tr_x)
the_test_prediction, _	= the_NB.predict(X=te_x)

my_train_conf_mat	= confusion_matrix(actual=tr_y, predicted=my_train_prediction)
my_test_conf_mat	= confusion_matrix(actual=te_y, predicted=my_test_prediction)
the_train_conf_mat	= confusion_matrix(actual=tr_y, predicted=the_train_prediction)
the_test_conf_mat	= confusion_matrix(actual_te_y, predicted=the_test_prediction)

my_train_metrics	= get_metrics(['accuracy', 'f1score'], my_train_conf_mat)
my_test_metrics		= get_metrics(['accuracy', 'f1score'], my_test_conf_mat)
the_train_metrics	= get_metrics(['accuracy', 'f1score'], the_train_conf_mat)
the_test_metrics	= get_metrics(['accuracy', 'f1score'], the_test_conf_mat)
