import numpy as np
import torch
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
def r_map(tops, labels, weight=None):
    map_ = []
    for instance_idx, (top, label) in enumerate(zip(tops, labels)):
        m = 0.
        relative_num = 0.
        for i, k in enumerate(top,start=1):
            if k == label:
                m += (relative_num + 1) / i
                relative_num += 1
        if relative_num > 0:
            m /= relative_num
        if weight: m *= weight[instance_idx]
        map_.append(m)
    return np.mean(map_)

def r_precision(tops, labels, weight=None):
    res = []
    tops = tops.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    for instance_idx, (ps, l) in enumerate(zip(tops, labels)):
        # ipdb.set_trace()
        showup = np.sum(ps == l)
        prec = showup / len(ps)
        if weight: prec *= weight[instance_idx]
        res.append(prec)
    return np.mean(res)

def r_acc(predict, label):
    return torch.sum(predict == label) / label.size(0)

def r_f1(predict, label, avg):
    return f1_score(label.cpu(), predict.cpu(), average=avg)

def weight_func(x):
    return np.cos(np.pi / 2 * x * 10)
