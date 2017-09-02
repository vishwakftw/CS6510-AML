# Main file supporting Question 1 Part 3 and 4
import KNN
import numpy as np
from time import clock
import dataset_gen as dg
from matplotlib import pyplot as plt
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

my_accuracies		= []
the_accuracies		= []
the_accuracies_2	= []

my_times	= []
the_times	= []
the_times_2	= []

max_k		= 21
for k in range(1, max_k+1):

	print('K = {0}'.format(k))
	# My KNN implementation
	my_knn		= KNN.KNN(K=k)
	my_knn.fit(X=tr_x, Y=tr_y)

	t_0			= clock()
	my_train_prediction	= my_knn.predict(X=tr_x)
	my_train_prdct_time	= round(clock() - t_0, 5)
	
	t_0			= clock()
	my_test_prediction	= my_knn.predict(X=te_x)
	my_test_prdct_time	= round(clock() - t_0, 5)

	# SKLearn KNN implementation with brute algorithm
	the_knn		= KNC(n_neighbors=k, algorithm='brute')
	the_knn.fit(X=tr_x, y=tr_y)

	t_0			= clock()
	the_train_prediction	= the_knn.predict(X=tr_x)
	the_train_prdct_time	= round(clock() - t_0, 5)

	t_0			= clock()
	the_test_prediction	= the_knn.predict(X=te_x)
	the_test_prdct_time	= round(clock() - t_0, 5)
	
	# SKLearn KNN implementation with auto algorithm
	the_knn_2	= KNC(n_neighbors=k)
	the_knn_2.fit(X=tr_x, y=tr_y)

	t_0			= clock()
	the_train_prediction_2	= the_knn_2.predict(X=tr_x)
	the_train_prdct_time_2	= round(clock() - t_0, 5)
	
	t_0			= clock()
	the_test_prediction_2	= the_knn_2.predict(X=te_x)
	the_test_prdct_time_2	= round(clock() - t_0, 5)

	# Get some information : train and test accuracies and times taken
	my_train_accuracy	= accuracy(actual=tr_y, predicted=my_train_prediction)
	the_train_accuracy	= accuracy(actual=tr_y, predicted=the_train_prediction)
	the_train_accuracy_2	= accuracy(actual=tr_y, predicted=the_train_prediction_2)

	my_test_accuracy	= accuracy(actual=te_y, predicted=my_test_prediction)
	the_test_accuracy	= accuracy(actual=te_y, predicted=the_test_prediction)
	the_test_accuracy_2	= accuracy(actual=te_y, predicted=the_test_prediction_2)

	# Store for graphing purposes	
	my_accuracies.append([my_train_accuracy, my_test_accuracy])
	the_accuracies.append([the_train_accuracy, the_test_accuracy])
	the_accuracies_2.append([the_train_accuracy_2, the_test_accuracy_2])
	
	my_times.append([my_train_prdct_time, my_test_prdct_time])
	the_times.append([the_train_prdct_time, the_test_prdct_time])
	the_times_2.append([the_train_prdct_time_2, the_test_prdct_time_2])

# Plotting the accuracies for different methods
plt.subplot(121)
plt.title('Train accuracies for different methods')
plt.xlabel('$k$', size=20)
plt.ylabel('Train Accuracy', size=20)
plt.plot(range(1, max_k+1), np.array(my_accuracies)[:,0], label='My Implementation')
plt.plot(range(1, max_k+1), np.array(the_accuracies)[:,0], label='sklearn with \'brute\'')
plt.plot(range(1, max_k+1), np.array(the_accuracies_2)[:,0], label='sklearn with \'auto\'')
plt.legend(loc='best')

plt.subplot(122)
plt.title('Test accuracies for different methods')
plt.xlabel('$k$', size=20)
plt.ylabel('Test Accuracy', size=20)
plt.plot(range(1, max_k+1), np.array(my_accuracies)[:,1], label='My Implementation')
plt.plot(range(1, max_k+1), np.array(the_accuracies)[:,1], label='sklearn with \'brute\'')
plt.plot(range(1, max_k+1), np.array(the_accuracies_2)[:,1], label='sklearn with \'auto\'')
plt.legend(loc='best')

plt.show()

# Plotting the times for different methods
plt.subplot(121)
plt.title('Time taken to get accuracies for different methods')
plt.xlabel('$k$', size=20)
plt.ylabel('Time (s)', size=20)
plt.yscale('log')
plt.plot(range(1, max_k+1), np.array(my_times)[:,0], label='My Implementation')
plt.plot(range(1, max_k+1), np.array(the_times)[:,0], label='sklearn with \'brute\'')
plt.plot(range(1, max_k+1), np.array(the_times_2)[:,0], label='sklearn with \'auto\'')
plt.legend(loc='best')

plt.subplot(122)
plt.title('Time taken to get accuracies for different methods')
plt.xlabel('$k$', size=20)
plt.ylabel('Time (s)', size=20)
plt.yscale('log')
plt.plot(range(1, max_k+1), np.array(my_times)[:,1], label='My Implementation')
plt.plot(range(1, max_k+1), np.array(the_times)[:,1], label='sklearn with \'brute\'')
plt.plot(range(1, max_k+1), np.array(the_times_2)[:,1], label='sklearn with \'auto\'')
plt.legend(loc='best')

plt.show()
