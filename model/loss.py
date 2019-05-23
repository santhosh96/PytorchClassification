import torch.nn.functional as F

def crossentropyloss(output, target):
    return F.cross_entropy(output, target)
