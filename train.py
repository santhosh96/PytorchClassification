import os
import json
import argparse
import torch
import pickle

import data_loader.data_extraction as dataextractor
import data_loader.data_loaders as module_data

import model.vgg19model as module_arch_vgg
import model.inceptionv3model as module_arch_inception
import model.alexnetmodel as module_arch_alexnet
import model.test_model as module_arch_testnet

import model.loss as module_loss
import model.metric as module_metric
import model.class_wise_metric as class_metric

from trainer import Trainer
from utils import Logger

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    
    # creating an object of logger
    train_logger = Logger()
    
    # creating an object of datahandling module, for generating the dataset in required format
    dataextract = get_instance(dataextractor, 'dataextract', config)
    # labelled, unlabelled and validation data, stored as numpy array
    training = dataextract.createdata()
    
    # creating data_loader instances
    # data_loaders.py is imported as module_data
    data_loader = get_instance(module_data, 'trainloader', config)
    valid_data_loader = data_loader.split_validation()

    # creating model architecture instance
    # architecture stated in config file is imported as model_instance
    if config['arch']['type'] == 'InceptionV3':
        model_instance = get_instance(module_arch_inception, 'arch', config)
    if config['arch']['type'] == 'VGG19Model':
        model_instance = get_instance(module_arch_vgg, 'arch', config)
    if config['arch']['type'] == 'AlexNet':
        model_instance = get_instance(module_arch_alexnet, 'arch', config)
    if config['arch']['type'] == 'Net':
        model = get_instance(module_arch_testnet, 'arch', config)
    
    if config['arch']['type'] != 'Net':
        model = model_instance.build_model(config['arch']['args']['model_path'])
    
    # loss function
    loss = getattr(module_loss, config['loss'])
    
    # metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    class_wise = getattr(class_metric, config['class_metric'])
    
    # optimizer, learning_rate scheduler

    # assigning trainable parameters
    trainable_params = model.parameters()
    
    # creating an optimizer (SGD)
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    
    # creating a learning rate scheduler
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    
    # creating an object of Trainer module
    trainer = Trainer(model, loss, metrics, class_wise, optimizer, 
                      resume = resume,
                      config = config,
                      data_loader = data_loader,
                      valid_data_loader = valid_data_loader,
                      lr_scheduler = lr_scheduler,
                      train_logger = train_logger )
    
    # Creating M labelled: Model using labelled dataset
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    # if args has config, or config file is provided, then load it into a variable 'config' 
    if args.config:
        # load config file
        config = json.load(open(args.config))

    # if checkpoint file is given then load it into a variable 'config' 
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    
    # raise error if none of them are given
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    # if the gpu device number is given then
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device 
    
    main(config, args.resume)
