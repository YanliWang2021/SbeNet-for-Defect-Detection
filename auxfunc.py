import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn import metrics
import matplotlib.pyplot as plt



def compute_roc(label,prob):
    auc = roc_auc_score(label, prob)
    fpr, tpr, thresholds = metrics.roc_curve(label, prob, pos_label=1)

    J = tpr-fpr
    idx = np.argmax(J)
    best_thresholds = thresholds[idx]

    fig = plt.figure(1, figsize=(8, 6))
    ax = plt.subplot(111)
    ax.plot(fpr, tpr, c='red')
    ax.set_xlim((-0.1,1.1))
    ax.set_ylim((-0.1,1.1))
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    return {'AUC':auc,'FPR':fpr,'TPR':tpr,'Thresholds':best_thresholds,'Fig':fig}

def compute_ap(label,prob):
    ap = average_precision_score(label, prob)
    pre, rec, thresholds = metrics.precision_recall_curve(label, prob, pos_label=1)

    F1 = 2*pre*rec/(pre+rec)
    idx = np.argmax(F1)

    best_threshold = thresholds[idx]

    fig = plt.figure(2, figsize=(8, 6))
    ax = plt.subplot(111)
    ax.plot(rec, pre, c='green')
    ax.set_xlim((-0.1,1.1))
    ax.set_ylim((-0.1,1.1))
    ax.set_xlabel('Pre')
    ax.set_ylabel('Rec')

    return {'AP':ap,'PRE':pre,'REC':rec,'Thresholds':best_threshold,'Fig':fig}

def setup_seed(seed=2021):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    y = np.array([0, 0, 1, 1])   #实际值
    scores = np.array([0.1, 0.4, 0.35, 0.8])  #预测值

    auc,fig = compute_roc(y,scores)
    fig.savefig('temp.jpg')
