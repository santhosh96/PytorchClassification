import torch


def overall_acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top3_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# def class_acc(output, target):
#     with torch.no_grad():
        
#         class_correct = list(0. for i in range(5))
#         class_total = list(0. for i in range(5))
        
#         _, preds = torch.max(output, 1)
#         assert preds.shape[0] == len(target)

#         corr = (preds == target).squeeze()
        
#         for i in range(len(preds)):
#             label = target[i]
#             class_correct[label] += corr[i].item()
#             class_total[label] += 1

#     return np.array(class_correct) / np.array(class_total)
