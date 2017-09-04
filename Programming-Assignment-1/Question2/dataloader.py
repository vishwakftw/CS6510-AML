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
	
def get_data(root, split=True):
	"""
		Function to get the data from all the files after vectorization
		Args:
			root		= Root directory of the EmailsData folder
			split		= (Optional, default = True) Requirement of split data or combined data
		Returns:
			numpy.ndarray(s) of shape (n_files, n_attributes), 
			where n_attributes is obtained using TfidfVectorizer in sklearn
			If split is true, then two numpy.ndarrays, else one numpy.ndarray
			
	"""
	train_files	= get_files(root=root, train=True)
	test_files	= get_files(root=root, train=False)
	all_files	= train_files + test_files
	
	vectorizer	= vct(input='filename')
	all_data	= vectorizer.fit_transform(all_files)
	all_data	= all_data.toarray()
	
	if split == True:
		train_data	= all_data[0 : len(train_files)]
		test_data	= all_data[ len(train_files) : ]
		
		return train_data, test_data
	else:
		return all_data
