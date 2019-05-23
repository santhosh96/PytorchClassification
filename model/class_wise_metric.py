import torch
import numpy as np

def class_accuracy(output, target, classes):
    with torch.no_grad():
        
        class_correct = list(0. for i in range(classes))
        class_total = list(0. for i in range(classes))
        
        _, preds = torch.max(output, 1)
        assert preds.shape[0] == len(target)

        corr = (preds == target).squeeze()
        
        for i in range(len(preds)):
            label = target[i]
            class_correct[label] += corr[i].item()
            class_total[label] += 1

    return [np.array(class_correct), np.array(class_total)]