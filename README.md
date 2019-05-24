# Pytorch Classification
A generalized structure for building image classification models using Pytorch
> This repository is extension and modification of [pytorch-template] (https://github.com/victoresque/pytorch-template)

## Getting Started

Please follow the instructions given below for setting up the environment

### Prerequisites

Following are the packages and libraries required for running the model

```
numpy >= 1.16.3
opencv-python >= 4.1.0.25
pandas >= 0.24.2
scikit-learn >= 0.21.1
torch >= 1.1.0
torchvision >= 0.3.0
```

### Setting up the environment (Optional)

The libraries and packages mentioned in pre-requisities can be installed directly, but creating a virtual environment is recommended

* Command for installing virtual environment:
```
pip3 install virtualenv
```
* Command for creating and activating the virtual environment:
```
virtualenv pytorchclassification
source pytorchclassification/bin/activate
```
* Command for deactivating the virtual environment
```
deactivate
```

### Installing the packages

The required packages and libraries can be installed by following command
```
pip3 install -r requirements.txt
```

## Folder Structure

```
  PytorchClassification/
  │
  ├── train.py - main script to start training
  ├── test.py - testing of trained model
  │
  ├── config/
  │   ├── alexnet_config.json -> config file for training the model using alexnet
  │   └── vgg_config.json -> config file for training the model using vgg19
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py -> base class for data loader
  │   ├── base_model.py -> base class for model
  │   └── base_trainer.py -> base class for model training
  │
  ├── data_loader/
  │   ├── data_extraction.py -> extract data from .pth files (CIFAR10 dataset)
  │   ├── data_handling.py -> few datahandling such as data extraction, splitting, randomizing
  │   ├── data_loaders.py -> data loader module
  │   └── image_dataset.py -> dataset module
  │
  ├── data/ -> default directory for storing input data
  │
  ├── model/ -> models, losses, and metrics
  │   ├── alexnetmodel.py -> model definition of alexnet
  |   ├── vgg19model.py -> model definition of vgg19  
  │   ├── metric.py -> metrics for overall accuracy and topk accuracy
  │   ├── class_wise_metric.py -> class wise accuracy
  │   └── loss.py -> loss function to be used
  │
  ├── results/ -> for saving results, classification reports
  │
  ├── saved/
  │   └── models/ - trained models are saved here
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │  
  └── utils/ - small utility functions
      ├── util.py -> utility for ensuring a given directory is present
      └── logger.py -> module for logging
```
## Usage

#### Training the model
```
python3 train.py -c config/alexnet_config.json
```
#### Testing the model
```
python3 test.py -r saved/path_to/best_model.pth
```
## Config file description
```
{
    "name": "CIFAR10",                                // Name of the training session
    "n_gpu": 1,                                       // Number of gpus to be used
    
    "dataextract": {                                  // For extracting the data from .pth
        "type": "Extractor",    
        "args": {
            "root": "data",                           // root folder of the saved data
            "base": "cifar-10-batches-py",            // base folder of the saved data
            "datalist": ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]  // data to be extracted
        }
    },
    
    "trainloader": {                                  // dataloader for training
        "type": "DataLoader",                         // selecting data loader
        "args": {
            "data_dir": "data/cifar-10-batches-py",   // dataset path
            "file": "training.pickle",                // file name 
            "batch_size": 32,                         // batch size
            "shuffle": true,                          // shuffle training data before splitting
            "validation_split": 0.2,                  // size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 4,                         // number of workers
            "input_size": 224                         // input size transformation
        }
    },
    
    "testloader": {                                   // dataloader for testing
        "type": "DataLoader",                         // selecting data loader
        "args": {
            "data_dir": "data/cifar-10-batches-py",   // dataset path   
            "file": "test_batch",                     // file name 
            "batch_size": 32,                         // batch size
            "shuffle": true,                          // shuffle training data before splitting
            "validation_split": 0,                    // size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 4,                         // number of workers
            "input_size": 224                         // input size transformation
        }
    },
    
    "arch": {                                         
        "type": "AlexNet",                            // model architecture type
        "args": {
            "trained": true,                          // trained or not
            "classes": 10,                            // number of classes in training data
            "model_path": "../../models/"             // path of the model
        }
    },
    
    "loss": "crossentropyloss",                       // loss function
    
    "metrics": [
        "overall_acc", "top3_acc"                     // metrics array
    ],
    
    "class_metric": "class_accuracy",                 // additional class wise metrics
    
    "optimizer": {
        "type": "SGD",
        "args":{
            "lr": 0.001,                              // learning rate
            "momentum": 0.9                           // momentum
        }
    },
    
    "lr_scheduler": {
        "type": "StepLR",                             // learning rate scheduler
        "args": {
            "step_size": 7,
            "gamma": 0.1
        }
    },
    
    "trainer": {
        "epochs": 50,                               // number of training epochs
        "save_dir": "saved/",                       // directory for saving the model
        "save_period": 1,                           // save checkpoints every save_period epochs
        "verbosity": 2,                             // 0: quiet, 1: per epoch, 2: full
        
        "monitor": "min val_loss",                  // mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 30,                           // number of epochs to wait before early stop. set 0 to disable.
        
        "tensorboardX": true,                       // tensorboardX is disabled
        "log_dir": "saved/runs"                     // saving directory of logs
    }

}
```
