from torch.utils.data import Dataset
import cv2
import os
import sys
import pickle
import numpy as np
from PIL import Image

def filereader(datapath,file):
	file_path = os.path.join(datapath, file)
	with open(file_path, 'rb') as f:
		# for handling the encoding
		if sys.version_info[0] == 2:
			entry = pickle.load(f)
		else:
			entry = pickle.load(f, encoding='latin1')
			
		data = entry['data']
		
		if 'target' in entry:
			target = entry['target']
				
	# reshaping the numpy array
	data = np.vstack(data).reshape(-1, 3, 32, 32)
	# transposing the array with the format of (32x32x3)
	data = data.transpose((0, 2, 3, 1))
	# returning the zip od data and target
	if 'target' in entry:
		return [data.astype('uint8'), np.array(target)]
	return [data.astype('uint8')]

class ImageDataset(Dataset):
	
	def __init__(self, datapath, file, transform=None):
		
		self.datapath = datapath
		self.file = file
		self.transform = transform
		self.data = filereader(self.datapath,self.file)
	
	def __len__(self):
		
		return len(self.data[0])
	
	def __getitem__(self, idx):
		
		image = self.data[0][idx]
		image = Image.fromarray(image)
        
		if len(self.data) > 1:
			label = self.data[1][idx]
		
		if self.transform:
			image = self.transform(image)
			
		if len(self.data) > 1:
			data = [image, label]
		else:
			data = [image]
		
		return data