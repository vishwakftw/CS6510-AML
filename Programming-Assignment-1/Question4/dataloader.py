import os
import numpy as np

def make_dict(cur_dict, key, values):
	"""
		Function to make a dictionary for the textual input
		Args:
			cur_dict	= Pre-existing dictionary object
			key		= Name of the key
			values		= Values to be taken within that key
		Returns:
			dictionary object
	"""
	cur_dict[key] = {}
	for i, v in enumerate(values):
		if v.find('?') != -1:
			cur_dict[key][v] = -1
		else:
			cur_dict[key][v] = i
	return cur_dict
	
def normalize_to_snd(matrix):
	"""
		Function to normalize the tensor to 0-mean and 1-variance (Standard Normal Distribution)
		Args:
			matrix		= matrix to be normalized
		Returns:
			normalized matrix of the same shape as the original matrix
	"""
	means, stddvs	= np.mean(matrix, axis=1), np.std(matrix, axis=1)
	means, stddvs	= means.repeat(matrix.shape[1]), stddvs.repeat(matrix.shape[1])
	means, stddvs	= means.reshape(matrix.shape), stddvs.reshape(matrix.shape)
	return (matrix - means)/stddvs
	
def get_dataset(root, normalize):
	"""
		Function to make the dataset
		Args:
			root		= Root destination for the data folder
			normalize	= Option to normalize to the data
		Returns:
			2-tuple of numpy.ndarray of shape (n_points, n_features (+1)) (+1 for training - outputs are appended to each input)
			1st element is the training data, 2nd element is the testing data
	"""
	# Construct dict based on README, and same encoding for text will be used for train and test data
	# Just data preprocessing section
	cur_dict	= {}	
	ref_dict	= {'workclass': 1, 'education': 3, 'marital-status': 5, 'occupation': 6, 'relationship': 7, 'race': 8, 'sex': 9, 'native-country': 13}
	val_dict	= {
	
			   'workclass'		: [' Private', ' Self-emp-not-inc', ' Self-emp-inc', ' Federal-gov', ' Local-gov', ' State-gov', ' Without-pay', ' Never-worked', ' ?'],

			   'education'		: [' Bachelors', ' Some-college', ' 11th', ' HS-grad', ' Prof-school', ' Assoc-acdm', ' Assoc-voc', ' 9th', ' 7th-8th', ' 12th', ' Masters', ' 1st-4th', ' 10th', ' Doctorate', ' 5th-6th', ' Preschool', ' ?'],
			   
			   'marital-status'	: [' Married-civ-spouse', ' Divorced', ' Never-married', ' Separated', ' Widowed', ' Married-spouse-absent', ' Married-AF-spouse', ' ?'],
			   
			   'occupation'		: [' Tech-support', ' Craft-repair', ' Other-service', ' Sales', ' Exec-managerial', ' Prof-specialty', ' Handlers-cleaners', ' Machine-op-inspct', ' Adm-clerical', ' Farming-fishing', ' Transport-moving', ' Priv-house-serv', ' Protective-serv', ' Armed-Forces', ' ?'],

			   'relationship'	: [' Wife', ' Own-child', ' Husband', ' Not-in-family', ' Other-relative', ' Unmarried', ' ?'],

			   'race'		: [' White', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other', ' Black', ' ?'],

			   'sex'		: [' Female', ' Male', ' ?'],

			   'native-country'	: [' United-States', ' Cambodia', ' England', ' Puerto-Rico', ' Canada', ' Germany', ' Outlying-US(Guam-USVI-etc)', ' India', ' Japan', ' Greece', ' South', ' China', ' Cuba', ' Iran', ' Honduras', ' Philippines', ' Italy', ' Poland', ' Jamaica', ' Vietnam', ' Mexico', ' Portugal', ' Ireland', ' France', ' Dominican-Republic', ' Laos', ' Ecuador', ' Taiwan', ' Haiti', ' Columbia', ' Hungary', ' Guatemala', ' Nicaragua', ' Scotland', ' Thailand', ' Yugoslavia', ' El-Salvador', ' Trinadad&Tobago', ' Peru', ' Hong', ' Holand-Netherlands', ' ?']
			  }

	for key in ref_dict.keys():
		cur_dict	= make_dict(cur_dict, key, val_dict[key])
		
	cur_path	= os.path.join(root, 'data')
	
	# Get train dataset file
	train_path	= os.path.join(cur_path, 'train.csv')
	train_data	= np.genfromtxt(train_path, delimiter=',', dtype=str)

	# Replace words with values in training data
	for i in range(0, train_data.shape[0]):
		for key in ref_dict.keys():
			train_data[i, ref_dict[key]] 	= cur_dict[key][train_data[i, ref_dict[key]]]
	
	# Get test dataset file
	test_path	= os.path.join(cur_path, 'test.csv')
	test_data	= np.genfromtxt(test_path, delimiter=',', dtype=str, usecols=range(0, 14))
	
	# Replace words with values in testing data
	for i in range(0, test_data.shape[0]):
		for key in ref_dict.keys():
			test_data[i, ref_dict[key]]	= cur_dict[key][test_data[i, ref_dict[key]]]
			
	train_data	= train_data.astype(float)
	test_data	= test_data.astype(float)
		
	assert np.isnan(train_data).any() == False, "Some conversions in Training Data have been missed"
	assert np.isnan(train_data).any() == False, "Some conversions in Testing Data have been missed"
	
	if normalize == True:
		# Normalize to standard normal
		train_data[:, :train_data.shape[1] - 1]	= normalize_to_snd(train_data[:, :train_data.shape[1] - 1])

		# Normalize to standard normal
		test_data	= normalize_to_snd(test_data)
		
	return train_data, test_data
