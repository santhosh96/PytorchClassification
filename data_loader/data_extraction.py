import os
import pickle

import data_loader.data_handling as handler

class Extractor:
    
    def __init__(self, root, base, datalist):
        self.root = root
        self.base = base
        self.datalist = datalist
    
    def createdata(self):
        # creating object for Datahandling class
        h = handler.DataHandling()

        # creating the data and target for training and validation set by calling data_extract function
        data, target = h.data_extract(self.root, self.base, self.datalist)

        combined_data = {
            'data' : data,
            'target' : target
        }

        file_name = os.path.join(self.root, self.base, 'training.pickle')
        
        with open(file_name, 'wb') as handle:
            pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        return combined_data