import numpy as np

def get_random(n_points, n_params, param_range=[0, 100], class_range=[0, 1]):
	"""
		Function to generate a dataset with n_points datapoints, with each input datapoint having n_params attributes.
		Args:
			n_points	:	The number of datapoints
			n_params	:	The number of attributes per input datapoint
			param_range	:	(Optional, default = [0, 100]) The range of the values taken by an attribute
			class_range	:	(Optional, default = [0, 1]) The discrete class values range
		Returns:
			numpy.ndarray of shape [n_points, n_params + 1]
	"""
	X	= np.random.randint(low=param_range[0],	high=param_range[1] + 1, size=(n_points, n_params))
	Y	= np.random.randint(low=class_range[0], high=class_range[1] + 1, size=(n_points)).reshape((n_points, 1))
	
	D	= np.concatenate((X, Y), axis=1)
	return D
	
def get_split(dset, train_split=0.8):
	"""
		Function to generate a train-test split based on the ratio.
		Args:
			dataset		: 	The numpy.ndarray
			train_split	: 	(Optional, default = 0.8) The Train-Test ratio
		Returns:
			(numpy.ndarray, numpy.ndarray) representing the train and test split respectively
	"""
	n_tr	= int(train_split*len(dset))
	tr_dset	= dset[ : n_tr]
	te_dset	= dset[n_tr : ]
	return tr_dset, te_dset
	
def get_dataset(n_pts, n_params, prm_rnge=[0, 100], clss_rnge=[0, 1], tr_splt=0.8, n_shffle=3, save_dst=True, save_splts=True):
	"""
		Function to generate a dataset with n_points datapoints, with each input datapoint having n_params attributes.
		This function will also create the splits for the train and test.
		Args:
			n_pts		:	Number of Datapoints
			n_params	: 	Number of attributes per input datapoint
			prm_rnge	:	(Optional, default = [0, 100]) Range of values for the attributes
			clss_rnge	:	(Optional, default = [0, 1]) The discrete class values range
			tr_splt		:	(Optional, default = 0.8) Train split ratio
			n_shffle	: 	(Optional, default = 3) Number of times to shuffle before splitting into train and test
			save_dst	: 	(Optional, default = True) Option for saving the original dataset in .csv file
			save_splts	: 	(Optional, default = True) Option for saving the splits separately in .csv files
	"""
	n_tr	= int(tr_splt*n_pts)
	dset	= get_random(n_points=n_pts, n_params=n_params, param_range=prm_rnge, class_range=clss_rnge)
	
	for i in range(0, n_shffle):
		np.random.shuffle(dset)
	
	indxs	= np.arange(1, n_pts + 1, 1).reshape((n_pts, 1))
	dset	= np.concatenate((indxs, dset), axis=1)
	
	tr_dset, te_dset	= get_split(dset)
	if save_dst == True:
		np.savetxt("dataset.csv", dset, delimiter=',', fmt='%d')
		
	if save_splts == True:
		np.savetxt("train_split.csv", tr_dset, delimiter=',', fmt='%d')
		np.savetxt("test_split.csv", te_dset, delimiter=',', fmt='%d')	
