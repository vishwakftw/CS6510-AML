import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as vct

def get_files(root, train=True):
	"""
		Function to get the filenames of all the files
		Args:
			root		= Root directory of the EmailsData folder
			train		= (Optional, default = True) Training / Testing files
		Returns:
			list of filenames
	"""
	cur_path	= os.path.join(root, 'EmailsData')
	file_type	= 'train' if train == True else 'test'
	
	ret_files	= []
	for fldr in os.listdir(cur_path):
		if fldr.find(file_type) != -1:
			req_path	= os.path.join(cur_path, fldr)
			for fl in os.listdir(req_path):
				ret_files.append(os.path.join(req_path, fl))
				
	return ret_files
	
def get_dataset(root, n_shuffle=3):
	"""
		Function to get the dataset from all the files after vectorization
		Args:
			root		= Root directory of the EmailsData folder
			n_shuffle	= (Optional, default = 3) Number of times to shuffle the data
		Returns:
			numpy.ndarrays of shape (n_files, n_attributes + 1) (train and test), 
			where n_attributes is obtained using TfidfVectorizer in sklearn
			
	"""
	train_files	= get_files(root=root, train=True)
	test_files	= get_files(root=root, train=False)
	all_files	= train_files + test_files
	
	vectorizer	= vct(input='filename')
	all_data	= vectorizer.fit_transform(all_files)
	all_data	= all_data.toarray()
	
	train_input	= all_data[ : len(train_files) ]
	test_input	= all_data[ len(train_files) : ]
	
	train_output	= []
	test_output	= []
	for f in all_files:
		if f.find('nonspam') != -1:
			if f.find('train') != -1:
				train_output.append(0)
			else:
				test_output.append(0)
		else:
			if f.find('train') != -1:
				train_output.append(1)
			else:
				test_output.append(1)
	
	assert len(train_output) == len(train_files) and len(test_output) == len(test_files), "Some issue with classification"
	train_output	= np.array(train_output).reshape((len(train_output), 1))
	test_output	= np.array(test_output).reshape((len(test_output), 1))
	
	train_set	= np.concatenate((train_input, train_output), axis=1)
	test_set	= np.concatenate((test_input, test_output), axis=1)
	
	for _ in range(0, n_shuffle):
		train_set	= np.random.permutation(train_set)
		test_set	= np.random.permutation(test_set)
		
	return train_set, test_set
