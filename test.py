import os
import argparse
import torch
import json
from tqdm import tqdm

import data_loader.data_loaders as module_data

import model.loss as module_loss
import model.metric as module_metric

import model.vgg19model as module_arch_vgg
import model.inceptionv3model as module_arch_inception
import model.alexnetmodel as module_arch_alexnet
import model.test_model as module_arch_testnet

from utils.util import ensure_dir

from train import get_instance
from sklearn.metrics import classification_report

def main(config, resume, target_class):
    
    # setup data_loader instances
    data_loader = getattr(module_data, config['testloader']['type'])(
        config['testloader']['args']['data_dir'],
        config['testloader']['args']['file'],
        batch_size=32,
        shuffle=True,
        validation_split=0.0,
        input_size=config['testloader']['args']['input_size'],
        num_workers=4
    )
    
    # build model architecture
    
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

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    y_pred = torch.tensor([],dtype=torch.long)
    y_true = torch.tensor([],dtype=torch.long)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = torch.argmax(output, dim=1)
                
            y_pred = torch.cat((y_pred, pred.cpu()))
            y_true = torch.cat((y_true, target.cpu()))
            
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    
    if target_class == None:
        target_names = [str(i) for i in range(0,config['arch']['args']['classes'])]
    else:
        target_names = target_class['targets']
    
    cl_report = classification_report(y_true, y_pred, target_names=target_names)
    
    ensure_dir('results')
    
    file_name = os.path.join('results', config['arch']['type']+'_classification_report.txt')
    
    accuracy = torch.sum(y_pred == y_true).item() / len(y_true)
    
    print('\nAccuracy of the model : ',accuracy)
    
    with open(file_name,'w') as fh:
        fh.writelines(cl_report)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--target', default=None, type=str,
                           help='name of the classes of prediction (default: indices in integer from 0 to classes-1 (provided in config file))')

    args = parser.parse_args()
    
    target_class = None
    
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    if args.target:
        target_class = json.load(open(args.target))

    main(config, args.resume, target_class)
