# Main file supporting Question 1 Part 3 and 4
import KNN
from time import clock
import numpy as np
import dataset_gen as dg
from sklearn.neighbors import KNeighborsClassifier as KNC

def accuracy(actual, predicted):
	assert len(actual) == len(predicted), "Wrong Dimensions of Labels"
	diffs	= np.abs(actual - predicted).mean()
	accrcy	= 1 - diffs
	
	return round(accrcy, 5)

# Generate dataset
dg.get_dataset(1500, 6)

# Get dataset from file
tr	= np.genfromtxt('train_split.csv', delimiter=',')
te	= np.genfromtxt('test_split.csv', delimiter=',')

tr_x, tr_y	= tr[: , 1:7], tr[: , 7]
te_x, te_y	= te[: , 1:7], te[: , 7]

for k in range(1, 22):
	# My KNN implementation
	my_knn	= KNN.KNN(K=k)
	my_knn.fit(X=tr_x, Y=tr_y)
	t_0			= clock()
	my_train_prediction	= my_knn.predict(X=tr_x)
	my_train_prdct_time	= round(clock() - t_0, 5)
	
	t_0			= clock()
	my_test_prediction	= my_knn.predict(X=te_x)
	my_test_prdct_time	= round(clock() - t_0, 5)

	# SKLearn KNN implementation
	the_knn	= KNC(n_neighbors=k, algorithm='brute')
	the_knn.fit(X=tr_x, y=tr_y)
	t_0			= clock()
	the_train_prediction	= the_knn.predict(X=tr_x)
	the_train_prdct_time	= round(clock() - t_0, 5)

	t_0			= clock()
	the_test_prediction	= the_knn.predict(X=te_x)
	the_test_prdct_time	= round(clock() - t_0, 5)

	# Get some information
	my_train_accuracy	= accuracy(actual=tr_y, predicted=my_train_prediction)
	the_train_accuracy	= accuracy(actual=tr_y, predicted=the_train_prediction)

	my_test_accuracy	= accuracy(actual=te_y, predicted=my_test_prediction)
	the_test_accuracy	= accuracy(actual=te_y, predicted=the_test_prediction)

	print('K = {0}\nMy Train Acc:\t{1}\tsklearn Train Acc:\t{2}\
	\nMy Test Acc:\t{3}\tsklearn Test Acc:\t{4}'.format(k, my_train_accuracy, the_train_accuracy, my_test_accuracy, the_test_accuracy))

	print('K = {0}\nMy Train Time:\t{1}s\tsklearn Train Time:\t{2}s\
	\nMy Test Time:\t{3}s\tsklearn Test Time:\t{4}s'.format(k, my_train_prdct_time, the_train_prdct_time, my_test_prdct_time, the_test_prdct_time))
