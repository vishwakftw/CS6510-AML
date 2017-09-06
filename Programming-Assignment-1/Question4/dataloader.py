import os
import numpy as np

def make_dict(cur_dict, key, values):
	cur_dict[key] = {}
	for i, v in enumerate(values):
		cur_dict[key][v] = i
	return cur_dict
	
def make_dataset(root):
	cur_path	= os.path.join(root, 'data')
	
	train_path	= os.path.join(cur_path, 'train.csv')
	tr	= np.genfromtxt(train_path, delimiter=', ', dtype=str)
	cur_dict	= {}	
	ref_dict	= {'workclass': 1, 'education': 3, 'marital-status': 5, 'occupation': 6, 'relationship': 7, 'race': 8, 'sex': 9, 'native-country': 13}
	for key in ref_dict:
		cur_dict	= make_dict(cur_dict, key, list(set(tr[:,ref_dict[key]].tolist())))
		
	return cur_dict
