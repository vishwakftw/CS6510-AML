import NB
import sys
import numpy as np
from dataloader import get_dataset
from sklearn.naive_bayes import GaussianNB

def accuracy(actual, predicted):
	assert len(actual) == len(predicted), "Wrong dimensions of Labels"
	diffs	= np.abs(actual - predicted).mean()
	accrcy	= 1 - diffs
	
	return round(accrcy, 5)

if len(sys.argv[1:]) != 1:
	print("Give the root folder as command line argument")
	sys.exit(1)
	
# Get the dataset
tr, te	= get_dataset(root=sys.argv[1])
tr_x, tr_y	= tr[ : , 0 : tr.shape[1]-1], tr[ : , tr.shape[1]-1]
te_x, te_y	= te[ : , 0 : te.shape[1]-1], te[ : , te.shape[1]-1]

my_NB	= NB.NB()
my_NB.fit(X=tr_x, Y=tr_y)
my_train_prediction	= my_NB.predict(X=tr_x)
my_test_prediction	= my_NB.predict(X=te_x)

the_NB	= GaussianNB()
the_NB.fit(X=tr_x, y=tr_y)
the_train_prediction	= the_NB.predict(X=tr_x)
the_test_prediction	= the_NB.predict(X=te_x)

my_train_accuracy	= accuracy(tr_y, my_train_prediction)
my_test_accuracy	= accuracy(te_y, my_test_prediction)
the_train_accuracy	= accuracy(tr_y, the_train_prediction)
the_test_accuracy	= accuracy(te_y, the_test_prediction)

print('Train Accuracies: {0} --> My NB\t {1} --> The NB\nTest Accuracies: {2} --> My NB\t {3} --> The NB'.format(my_train_accuracy, the_train_accuracy, my_test_accuracy, the_test_accuracy))
