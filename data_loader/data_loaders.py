from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.image_dataset import ImageDataset


class DataLoader(BaseDataLoader):
    
    def __init__(self, data_dir, file, batch_size, shuffle, validation_split, num_workers, input_size):
        
        # transformations to be applied
        trsfm = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                ])
        
        # path of the text file that consists each image name and label
        self.data_dir = data_dir
        
        # creating the dataset object
        self.dataset = ImageDataset(self.data_dir, file, transform = trsfm)
        
        # passing dataset object along with other parameters 
        super(DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
    