import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from sklearn.metrics import classification_report


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, class_wise, optimizer, resume, config, data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        
        super(Trainer, self).__init__(model, loss, metrics, class_wise, optimizer, resume, config, train_logger)
        
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.data_loader.batch_size))

    def _eval_metrics(self, output, target):
        
        acc_metrics = np.zeros(len(self.metrics))
        
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            
            if self.config['arch']['type'] == 'InceptionV3':
                output, aux_output = self.model(data)
                loss1 = self.loss(output, target)
                loss2 = self.loss(aux_output, target)
                loss = loss1 + 0.4 * loss2
            
            else:
                output = self.model(data)
                loss = self.loss(output, target)
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)
        
            
        log = {
            'train_loss': total_loss / len(self.data_loader),
            'train_metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        
        class_wise_accurate = np.zeros(self.config['arch']['args']['classes'])
        class_wise_counts = np.zeros(self.config['arch']['args']['classes'])
        y_pred = torch.tensor([],dtype=torch.long)
        y_true = torch.tensor([],dtype=torch.long)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                
                pred = torch.argmax(output, dim=1)
                
                y_pred = torch.cat((y_pred, pred.cpu()))
                y_true = torch.cat((y_true, target.cpu()))
                
                loss = self.loss(output, target)
                
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                
                class_wise_metrics = self.class_wise(output, target, self.config['arch']['args']['classes'])
                class_wise_accurate = class_wise_accurate + class_wise_metrics[0]
                class_wise_counts = class_wise_counts + class_wise_metrics[1]
        
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
            'val_cl_acc': np.around(class_wise_accurate / class_wise_counts, decimals=3).tolist()
        }
