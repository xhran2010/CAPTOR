import os
import random
import numpy as np
import scipy.sparse as sp
import time

import pgl
import paddle

try:
    import ipdb
except:
    pass

def strfy_args(args):
    pass

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

class Device(object):
    def __init__(self, device_name):
        self.dv_name = device_name
    
    def transfer(self, x):
        x.to(self.dv_name)

def save_model(model, i, save_dir, optimizer=None, scheduler=None):
    """ save current model """
    paddle.save(model.state_dict(), os.path.join(save_dir, 'model_{}.xhr'.format(i)))

def path_exist(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def filt_params(named_params, filt_key):
    filted = []
    for name, par in named_params:
        if filt_key in name:
            filted.append(par)
    return filted

def delete_models(epochs, base_path):
    for e in epochs:
        os.remove(os.path.join(base_path, "model_{}.xhr".format(e)))

class Logger(object):
    def __init__(self, log_path, name, seed, is_write_file=True):
        cur_time = time.strftime("%m-%d-%H:%M", time.localtime())
        self.is_write_file = is_write_file
        if self.is_write_file:
            self.log_file = open(os.path.join(log_path, "%s %s(%d).log" % (cur_time, name, seed)), 'w')
    
    def log(self, log_str):
        out_str = "[%s] " % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + log_str
        print(out_str)
        if self.is_write_file:
            self.log_file.write(out_str+'\n')
            self.log_file.flush()
    
    def close_log(self):
        if self.is_write_file:
            self.log_file.close()

def checkin_graph_struct(o_ck):
    inputs = o_ck.cpu().numpy()
    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)
    for u_input in inputs:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0) # out degree of every u
        u_sum_in[np.where(u_sum_in == 0)] = 1 # avoid divided by zero
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) # re-index
    alias = torch.LongTensor(alias_inputs)
    A = torch.FloatTensor(A)
    items = torch.LongTensor(items)
    
    return alias, A, items


def eval_sampling(region_poi, batch_size, device, duplicate=False, k=10):
    if not duplicate:
        poi_samples, region_samples = [], []
        for r, v in region_poi.items():
            s = np.random.choice(list(v), (batch_size, k))
            poi_samples.append(torch.from_numpy(s))
            region_samples.append(torch.from_numpy(np.full((batch_size, k), r)))
        poi_samples = torch.cat(poi_samples, dim=1).to(device)
        region_samples = torch.cat(region_samples, dim=1).to(device)
        return poi_samples, region_samples
    else:
        poi_samples = []
        region_samples = []
        for r, v in region_poi.items():
            s = np.random.choice(list(v), (1, k))
            poi_samples.append(torch.from_numpy(s))
            region_samples.append(torch.from_numpy(np.full((1, k), r)))
        poi_samples = torch.cat(poi_samples, dim=1).to(device)
        region_samples = torch.cat(region_samples, dim=1).to(device)
        return poi_samples, region_samples

def collate_fn(batch):
    uid, ori_ck, dst_ck, ori_rg, dst_rg = zip(*batch)
    
    ori_max_len = np.max([i.shape[0] for i in ori_ck])
    dst_max_len = np.max([i.shape[0] for i in dst_ck])

    pad_ori_ck = [np.concatenate([i, np.zeros(ori_max_len - i.shape[0], dtype='int64')]) for i in ori_ck]
    pad_dst_ck = [np.concatenate([i, np.zeros(dst_max_len - i.shape[0], dtype='int64')]) for i in dst_ck]

    pad_ori_ck = paddle.to_tensor(pad_ori_ck)
    pad_dst_ck = paddle.to_tensor(pad_dst_ck)
    
    ori_rg = paddle.to_tensor(ori_rg)
    dst_rg = paddle.to_tensor(dst_rg)
    uid = paddle.to_tensor(uid)

    return uid, pad_ori_ck, pad_dst_ck, ori_rg, dst_rg

def normalize_ngcf(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_gat(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    ret = torch.sparse.FloatTensor(indices, values, shape)
    if cuda(): ret = ret.cuda()
    return ret

def read_graph(dir_path):
    poi_poi_graph = sp.load_npz(os.path.join(dir_path, 'poi_poi_graph_ada.npz'))
    region_poi_graph = sp.load_npz(os.path.join(dir_path, 'region_poi_graph.npz'))
    #pp_adj = sparse_mx_to_torch_sparse_tensor(poi_poi_graph)#.to_dense()
    #rp_adj = sparse_mx_to_torch_sparse_tensor(region_poi_graph)#.to_dense()
    #return pp_adj, rp_adj
    return poi_poi_graph, region_poi_graph

def sparse_to_pgl(adj):
    adj = adj.tocoo()
    row = adj.row
    col = adj.col
    edge_list = [(src, dst) for src, dst in zip(row, col)]
    graph = pgl.Graph(
        edges = edge_list,
        num_nodes = adj.shape[0]
    )
    return graph.tensor()

def subgraph(adj, select_indice):
    if type(select_indice) != list: 
        select_indice = select_indice.tolist()

    col_indice = [select_indice]
    row_indice = [[i] for i in select_indice]
    
    adj = adj.tolil()
    sub_adj = adj[row_indice, col_indice]
    sub_adj = sparse_to_pgl(sub_adj)

    return sub_adj

def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)

