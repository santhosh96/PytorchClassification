from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.image_dataset import ImageDataset


class DataLoader(BaseDataLoader):
    
    def __init__(self, data_dir, file, batch_size, shuffle, validation_split, num_workers, input_size):
        
        # transformations to be applied
        trsfm = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        
        # path of the text file that consists each image name and label
        self.data_dir = data_dir
        
        test = False
        
        # creating the dataset object
        self.dataset = ImageDataset(self.data_dir, file, test, transform = trsfm)
        
        # passing dataset object along with other parameters 
        super(DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class TestLoader(BaseDataLoader):
    
    def __init__(self, data_dir, file, batch_size, shuffle, validation_split, num_workers, input_size):
        
        # transformations to be applied
        trsfm = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        
        # path of the text file that consists each image name and label
        self.data_dir = data_dir
        
        test = True
        
        # creating the dataset object
        self.dataset = ImageDataset(self.data_dir, file, test, transform = trsfm)
        
        # passing dataset object along with other parameters 
        super(TestLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)