import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, f1_score


# ########### NODE CLASSIFICATION METRICS ###############


def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1


# ########### RECONSTRUCTION METRICS ###############


def mean_average_precision(adj, rec):
    nb_nodes = adj.shape[0]
    map_score = 0
    for i in range(nb_nodes):
        pred = np.array(list(rec[i, :i]) + list(rec[i, i + 1:]))
        gt = np.array(list(adj[i, :i]) + list(adj[i, i + 1:]))
        map_score += average_precision_score(gt, pred)
    map_score /= 1. * nb_nodes
    return map_score
