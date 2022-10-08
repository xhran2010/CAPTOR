import numpy as np
from sklearn.metrics import f1_score
try:
    import ipdb
except:
    pass

# POI Metrics
def p_rec(tops,labels,k):
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label)
        r = len(hit) / (len(set(label)) - 1) # 1 for '0'
        res += r
    return res

def p_precision(tops, labels, k):
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label)
        r = len(hit) / k # 1 for '0'
        res += r
    return res

def p_f1(tops, labels, k):
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label)
        p = len(hit) / k # 1 for '0'
        r = len(hit) / (len(set(label)) - 1) # 1 for '0'
        try:
            res += (2 * p * r / (p + r))
        except:
            res += 0
    return res

def p_ndcg(tops, labels, k):
    res = 0.
    for top, label in zip(tops, labels):
        dcg = 0.
        idcg = 0.
        for i, p in enumerate(top[:k], start=1):
            rel = 1 if np.isin(p, label) else 0
            dcg += (2 ** rel - 1) / (np.log2(i + 1))
            idcg += 1 / (np.log2(i + 1))
        ndcg = dcg / idcg
        res += ndcg
    return res

# Region Metrics
def r_acc(predict, label):
    label = label.numpy()
    return np.sum(predict == label) / label.shape[0]
