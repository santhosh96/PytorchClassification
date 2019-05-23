# Pytorch Classification
A generalized structure for building image classification models using Pytorch

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
