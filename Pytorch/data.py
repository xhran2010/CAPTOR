from random import shuffle, choice
import numpy as np
import scipy.sparse as sp
from copy import copy
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import torch
import json
import dgl
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

        """
        with open(os.path.join(feat_path, "static_feat.txt")) as f_basic_feat, \
            open(os.path.join(feat_path, "freq_feat.txt")) as f_freq_feat:
            basic_feat, freq_feat = {}, {}
            for l in f_basic_feat:
                l = json.loads(l.strip())
                basic_feat[l['cuid']] = l['feature']
            for l in f_freq_feat:
                l = json.loads(l.strip())
                freq_feat[l['cuid']] = l['feature']
        """
        
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

            """
            try:
                u_basic = basic_feat[cuid]
                u_basic = torch.FloatTensor(u_basic)
            except:
                u_basic = torch.zeros(len(list(basic_feat.values())[0]))
            try:
                u_freq = freq_feat[cuid]
                u_freq = torch.FloatTensor(u_freq)
            except:
                u_freq = torch.zeros(len(list(freq_feat.values())[0]))
            """
            
            #self.feats.append(torch.cat([u_basic, u_freq]))
            #self.feats.append(u_basic)
            self.uids.append(int(uid))
        
        #self.feats = torch.stack(self.feats, dim=0)
        #self.feats = self.feats.to(args.device)
    
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

        # NGCF data
        try:
            self.home_adj = sp.load_npz("./graph/home_adj.npz")
        except:
            #self.home_adj = self._create_home_adj(home_check_mat)
            #sp.save_npz("./graph/home_adj.npz", self.home_adj)
            self.home_adj = None

        # CRF data
        if args.crf:
            try:
                self.pp_adj = sp.load_npz(args.pp_graph_path)
                self.pp_adj = dgl.from_scipy(self.pp_adj)
                self.pp_adj = self.pp_adj.to(args.device)
            except:
                n_poi = len(self.poi_idx) + 1
                pp_graph = np.zeros((n_poi, n_poi))
                row, col = [0], [0]
                for items in self.region_poi.values():
                    for i in items:
                        row.extend([i] * len(items))
                        col.extend(items)
                data = [1] * len(row)
                self.pp_adj = sp.coo_matrix((data, (row, col)), shape=(n_poi, n_poi))
                self.pp_adj = normalize_gat(self.pp_adj)
                self.pp_adj = torch.FloatTensor(np.array(self.pp_adj.todense()))
                self.pp_adj = self.pp_adj.to(args.device)
        else:
            self.pp_adj = None
    
    # NGCF
    def _create_home_adj(self, R):
        n_users = len(self.trans)
        n_items = len(self.poi_idx) + 1

        adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()

        adj_mat[:n_users, n_users:] = R
        adj_mat[n_users:, :n_users] = R.T
        adj_mat = adj_mat.todok()

        def mean_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()
        
        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()
        
    def __getitem__(self, index):
        uid = self.uids[index]
        o = self.oris[index]
        d = self.dsts[index]
        t = self.trans[index]
        ori_ck = torch.LongTensor(list(map(lambda y: y[0], o)))
        dst_ck = torch.LongTensor(list(map(lambda y: y[0], d))).unique()
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
        poi_samples = torch.LongTensor(poi_samples).to(self.args.device)
        region_samples = torch.LongTensor(region_samples).to(self.args.device)
            
        return poi_samples, region_samples
    
    def region_mask(self):
        mask = torch.zeros((len(self.region_idx), len(self.poi_idx) + 1))
        for k, v in self.region_poi.items():
            mask[k, list(v)] = 1.
        mask = mask / torch.sum(mask, dim=-1, keepdim=True).expand_as(mask)
        mask = mask.to(self.args.device)

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
    
    # sample not by region
    """
    us_shuf = copy([i for i in range(len(trans))])
    np.random.shuffle(us_shuf)
    us_len = len(us_shuf)

    train_offset = int(us_len * ratios[0])
    valid_offset = int(us_len * (ratios[0] + ratios[1]))

    train_indice.extend(us_shuf[:train_offset])
    valid_indice.extend(us_shuf[train_offset:valid_offset])
    test_indice.extend(us_shuf[valid_offset:])
    """
    
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
    """
    n_region = len(dataset.region_idx)
    sub_trans = np.array(dataset.trans)[train_indice]
    adj = sp.coo_matrix((np.ones(sub_trans.shape[0]), (sub_trans[:, 0], sub_trans[:, 1])),
                        shape=(n_region + len(dataset.poi_idx) + 1, n_region + len(dataset.poi_idx) + 1),
                        dtype=np.float32)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    """
    return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice)
