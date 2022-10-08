from random import shuffle, choice
import numpy as np
import scipy.sparse as sp
from copy import copy
from collections import defaultdict
from paddle.io import Dataset, Subset
import paddle
import json
import pgl
import os


try:
    import ipdb
except:
    pass

from utils import *

class TravelDataset(Dataset):
    def __init__(self, args, ori_data_path, dst_data_path, trans_data_path, feat_path):
        ori_raw = list(map(lambda x: x.strip().split('\t'), open(ori_data_path, 'r')))
        dst_raw = list(map(lambda x: x.strip().split('\t'), open(dst_data_path, 'r')))
        trans_raw = list(map(lambda x: x.strip().split('\t'), open(trans_data_path, 'r')))
        self.args = args
        
        self.poi_idx = {}
        self.region_idx = {}
        self.tag_idx = {}
        self.region_poi = defaultdict(set)

        self.trans = []
        self.feats = []
        self.uids = []

        for i in trans_raw:
            uid, cuid, ori_region, dst_region = i
            if ori_region not in self.region_idx: 
                self.region_idx[ori_region] = len(self.region_idx) 
            if dst_region not in self.region_idx:
                self.region_idx[dst_region] = len(self.region_idx)
            self.trans.append((self.region_idx[ori_region], self.region_idx[dst_region]))
            
            self.uids.append(int(uid))
        
        for i in ori_raw + dst_raw:
            uid, cuid, _, bid, timestamp, std_tag = i
            #uid, timestamp, bid, std_tag = i
            if bid not in self.poi_idx:
                self.poi_idx[bid] = len(self.poi_idx) + 1
            if std_tag not in self.tag_idx:
                self.tag_idx[std_tag] = len(self.tag_idx)
        self.oris = []
        self.dsts = []
        
        ori_buffer = []
        dst_buffer = []

        #home_check_mat = sp.dok_matrix((len(self.trans), len(self.poi_idx) + 1), dtype=np.float32)

        last_uid = '0'
        for i in ori_raw:
            uid, cuid, _, bid, timestamp, std_tag = i
            self.region_poi[self.trans[int(uid)][0]].add(self.poi_idx[bid])
            if uid != last_uid:
                self.oris.append(ori_buffer)
                ori_buffer = []
                last_uid = uid
            ori_buffer.append((self.poi_idx[bid], self.tag_idx[std_tag], timestamp))
            #home_check_mat[int(uid), self.poi_idx[bid]] = 1.
        self.oris.append(ori_buffer)
        
        last_uid = '0'
        for i in dst_raw:
            uid, cuid, _, bid, timestamp, std_tag = i
            self.region_poi[self.trans[int(uid)][1]].add(self.poi_idx[bid])
            if uid != last_uid:
                self.dsts.append(dst_buffer)
                dst_buffer = []
                last_uid = uid
            dst_buffer.append((self.poi_idx[bid], self.tag_idx[std_tag], timestamp))
        self.dsts.append(dst_buffer)

        self.eval_samples = self._eval_sampling()
        self.home_adj = None

        # CRF data
        if args.crf:
            self.pp_adj = sp.load_npz(args.pp_graph_path).tocoo()
        else:
            self.pp_adj = None
    
    def __getitem__(self, index):
        uid = self.uids[index]
        o = self.oris[index]
        d = self.dsts[index]
        t = self.trans[index]
        ori_ck = np.array(list(map(lambda y: y[0], o)))
        dst_ck = np.unique(np.array(list(map(lambda y: y[0], d))))
        ori_rg = t[0]
        dst_rg = t[1]
        
        return uid, ori_ck, dst_ck, ori_rg, dst_rg
    
    def __len__(self):
        return len(self.trans)
    
    def _eval_sampling(self, k=10):
        poi_samples, region_samples = [], []
        for r, v in self.region_poi.items():
            s = np.random.choice(list(v), k)
            poi_samples.extend(s)
            region_samples.extend([r for _ in range(k)])
        poi_samples = paddle.to_tensor(poi_samples)
        region_samples = paddle.to_tensor(region_samples)
            
        return poi_samples, region_samples
    
    def region_mask(self):
        mask = paddle.zeros((len(self.region_idx), len(self.poi_idx) + 1))
        for k, v in self.region_poi.items():
            mask[k, list(v)] = 1.
        mask = mask / paddle.sum(mask, axis=-1, keepdim=True).expand_as(mask)

        return mask
    
    def get_proportion(self):
        dest_count = defaultdict(int)
        for _, d in self.trans:
            dest_count[d] += 1
        dest_prop = list()
        for i in range(len(self.region_idx)):
            dest_prop.append(dest_count[i] / len(self.trans))
        return dest_prop

def random_split(dataset, split_path, ratios=[0.8, 0.1, 0.1], ):
    trans = dataset.trans
    trans_by_pair = defaultdict(list)
    for u, t in enumerate(trans):
        trans_by_pair[t].append(u)
    
    train_indice, valid_indice, test_indice = [], [], []
    
    # sample by region
    #"""
    train_indice, valid_indice, test_indice = np.load(split_path, allow_pickle=True)

    """
    for t, us in trans_by_pair.items():
        us_shuf = copy(us)
        np.random.shuffle(us_shuf)
        us_len = len(us) # the length is required >= 10

        train_offset = int(us_len * ratios[0])
        valid_offset = int(us_len * (ratios[0] + ratios[1]))

        train_indice.extend(us_shuf[:train_offset])
        valid_indice.extend(us_shuf[train_offset:valid_offset])
        test_indice.extend(us_shuf[valid_offset:])
    """
    return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice)
