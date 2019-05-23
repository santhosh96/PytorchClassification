import pickle
import numpy as np
import pandas as pd
import os
import sys
import random
from PIL import Image

class DataHandling:

	def data_extract(self, root, base, data_list):
		'''
			Function for unpickling the dataset of CIFAR 10
			Input: folder names of root and base and a list of file names
			Output: Numpy array of data and list of target
		'''
		
		data = [None] * len(data_list)
		target = []

		for idx, file_name in enumerate(data_list):
			
			# complete filepath    
			file_path = os.path.join(root, base, file_name)
			
			with open(file_path, 'rb') as f:
				# for handling the encoding
				if sys.version_info[0] == 2:
					entry = pickle.load(f)
				else:
					entry = pickle.load(f, encoding='latin1')
				
				data[idx] = entry['data']
				
				if 'labels' in entry:
					target.extend(entry['labels'])
				else:
					target.extend(entry['fine_labels'])
		# reshaping the numpy array
		data = np.vstack(data).reshape(-1, 3, 32, 32)
		# transposing the array with the format of (32x32x3)
		data = data.transpose((0, 2, 3, 1))
		# returning the zip od data and target
		return (data.astype('uint8'), np.array(target))
	
	def data_randomize(self, data, target):
		'''
			Function for randomizing the dataset if required
			Input: dataset with the list of images array and target list separately
			Output: Shuffled dataset
		'''

		if type(data) != list:
			data = data.tolist()
		if type(target) != list:
			target = target.tolist()
		# zipping together d and t
		rand = list(zip(data, target))
		# randomizing the rand list
		random.shuffle(rand)
		
		return rand

	def data_split(self, data, target):
		'''
			Function for splitting the dataset into labelled and unlabelled dataset
			Input: dataset with the list of images array and target list separately
			Output: X_data, X_target, Y_data
		'''

		# list for data of labelled dataset
		X_data = []
		# list for target of of labelled dataset
		X_target = []
		# list for the chosen indices
		indices = np.array([], dtype=np.int32)

		# for each class of labels
		for label in range(len(np.unique(np.array(target)))):
			# choose indices of 1000 rows having target value of the label
			indx = np.where(target == label)[0][:1000]
			# appending the chosen index to the indices list
			indices = np.append(indices,indx)
			# for each index of the indx list, append the data to X_data and target to X_target
			for index in indx:
				X_data.append(data[index])
				X_target.append(target[index])
	   
		# randomizing the dataset    
		X_data, X_target = zip(*self.data_randomize(X_data, X_target))
		X_data = np.array(X_data).astype('uint8') 
		X_target = np.array(X_target)
		# deleting the data that has been assigned to labelled data from original dataset
		Y_data = np.delete(data, indices, axis=0)

		return (X_data, X_target, Y_data)