import datetime
import math
import numpy as np
import random
from utils import eval_sampling, subgraph
import sys

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl


try:
    import ipdb
    from tqdm import tqdm
except:
    pass

class Attention(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias_attr=False)
        self.sim_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.ReLU()
        )
    
    def forward(self, query, key, value): 
        # query: B x H, key, value: B x S x H
        query = self.W(query)
        key = self.W(key)
        value = self.W(value)
        score = F.softmax(self.sim_mlp(paddle.concat([query, key], axis=-1)), axis=-2) # B x S x 1
        result = paddle.sum(score * value, axis=-2)
        return result, score


class CRFLayer(nn.Layer):
    def __init__(self, hidden_size, alpha, beta, gamma=0.2, dropout=0.6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.hidden_size = hidden_size

        self.W_fc = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False)
        self.attn_fc = nn.Linear(2 * hidden_size, 1, bias_attr=False)
        self.leakyrelu = nn.LeakyReLU(self.gamma)

    def message_func(self, src_feat, dst_feat, edge_feat):
        # message UDF for equation (3) & (4)
        z2 = paddle.concat([src_feat['emb_attn'], dst_feat['emb_attn']], axis=1) # N x 2h
        a = self.attn_fc(z2)
        return {'z': src_feat['emb_crf'], 'e': self.leakyrelu(a)}
    
    def reduce_func(self, msg):
        alpha = msg.reduce_softmax(msg['e']) # N x 1
        # equation (4)
        feature = msg['z']
        feature = feature * alpha
        h = msg.reduce(feature, pool_type='sum')
        return h

    def forward(self, embedding_input, h_input, graph):
        z = self.W_fc(h_input)
        msg = graph.send(
            self.message_func,
            src_feat={
                'emb_attn': z,
                'emb_crf': h_input
            },
            dst_feat={
                'emb_attn': z
            }
        )
        crf_output = graph.recv(reduce_func=self.reduce_func, msg=msg)
        output = (self.alpha * embedding_input + self.beta * crf_output) / (self.alpha + self.beta)
        return output, msg


class CRF(nn.Layer):
    def __init__(self, args, alpha, beta, n_layer):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_layer = n_layer
        
        # self.layers = nn.ModuleList([CRFLayer(args.hidden_size, alpha, beta) for _ in range(self.n_layer)])
        self.layer = CRFLayer(args.hidden_size, alpha, beta)

    def message_func(self, src_feat, dst_feat, edge_feat):
        return {'z': src_feat['emb_crf'], 't': dst_feat['emb_crf']}
    
    def reduce_func(self, msg):
        alpha = msg.reduce_softmax(msg['e'])
        h = msg.reduce(alpha.squeeze(1) * paddle.norm(msg['t'] - msg['z'], p=2, axis=-1), pool_type='sum')
        h = h.unsqueeze(1)
        return h
    
    def forward(self, embedding_input, graph, is_train=True):
        for n in range(self.n_layer):
            if n == 0:
                h_input = embedding_input
            h_input, attn_msg = self.layer(embedding_input, h_input, graph)
        
        if is_train:
            loss_a = paddle.norm(h_input - embedding_input, 2, -1) ** 2

            msg = graph.send(
                self.message_func,
                src_feat = {
                    'emb_crf': h_input
                },
                dst_feat = {
                    'emb_crf': h_input
                }
            )
            msg['e'] = attn_msg['e']
            loss_b = graph.recv(reduce_func=self.reduce_func, msg=msg)
            loss = paddle.mean(self.alpha * loss_a + self.beta * loss_b)
        else:
            loss = None
        return h_input, loss


class Memory(nn.Layer):
    def __init__(self, args, n_region, n_slot, hidden_size):
        super().__init__()
        self.n_region = n_region
        self.n_slot = n_slot
        self.hidden_size = hidden_size
        self.args = args
        self.memory = nn.Embedding(n_region * n_slot, hidden_size)

        self.forget_linear = nn.Linear(hidden_size * 2, 1, bias_attr=False)
        self.attn = Attention(hidden_size)

    def get_memory(self):
        return self.memory.weight.reshape([self.n_region, self.n_slot, -1])
    
    def forward(self, memory, write_pack=None, read_pack=None, state="w"):
        if state == "w":
            o_emb, o_rg = write_pack
            select_mem = memory[o_rg] # B x S x H -> S x H
            o_emb_ext = o_emb.unsqueeze(0).expand([self.n_slot, -1])
            forget_gate = F.sigmoid(self.forget_linear(paddle.concat([o_emb_ext, select_mem], axis=-1))) # S x 1

            forget_pad = paddle.zeros([self.n_region, self.n_slot, 1])
            forget_pad[o_rg] = forget_gate

            new_mem = memory * (1 - forget_pad) + o_emb_ext * forget_pad

            return new_mem
        if state == "r":
            o_emb, d_rg = read_pack
            mem_out, attn_weight = self.attn(o_emb.unsqueeze(1).expand([-1, self.n_slot, -1]), memory[d_rg], memory[d_rg])
            return mem_out, attn_weight
        if state == "ra":
            o_emb, _ = read_pack
            mem_out, _ = self.attn(
                o_emb.reshape([-1, 1, 1, self.hidden_size]).expand([-1, self.n_region, self.n_slot, -1]),
                memory.unsqueeze(0).expand([o_emb.shape[0], -1, -1, -1]),
                memory.unsqueeze(0).expand([o_emb.shape[0], -1, -1, -1]))
            
            # mem_out = memory.mean(dim=1).unsqueeze(0).expand(o_emb.size(0), -1, -1)
            return mem_out
        

class Dyna(nn.Layer):
    def __init__(self, args, n_poi, n_region, eval_samples, region_poi, region_mask, pp_adj):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_poi = n_poi
        self.n_region = n_region
        self.margin = args.margin
        self.args = args
        
        self.pp_adj = pp_adj
        self.eval_samples = eval_samples # 0-poi, 1-region
        self.region_poi = region_poi
        self.region_mask = region_mask

        nn.initializer.set_global_initializer(nn.initializer.XavierUniform())

        self.poi_embedding = nn.Embedding(self.n_poi, self.hidden_size)
        if self.args.trans == 'transr':
            self.region_embedding = nn.Embedding(self.n_region, self.hidden_size * self.hidden_size)
        else:
            self.region_embedding = nn.Embedding(self.n_region, self.hidden_size)

        self.head_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.tail_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.crf = CRF(args, args.alpha, args.beta, args.crf_layer)

        # memory
        self.n_slot = args.mem_slot
        self.memory = Memory(args, self.n_region, self.n_slot, self.hidden_size)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )

        self.reg_attn = Attention(self.hidden_size)
    
        
    def _avg_pooling(self, ck, emb):
        emb_sum = paddle.sum(emb, axis=1)
        row_count = paddle.sum(ck != 0, axis=-1)
        emb_agg = emb_sum / row_count.unsqueeze(1).expand_as(emb_sum)
        return emb_agg
    
    def neg_sampling(self, k, d_ck, is_mask):
        neg_samples = []
        for idx, t in enumerate(d_ck):
            t = t.cpu().numpy()
            neg_sample = np.random.choice(np.setdiff1d(np.arange(self.n_poi), t), k)
            if is_mask: neg_sample = np.where(t == 0, 0, neg_sample)
            neg_samples.append(paddle.to_tensor(neg_sample))
        neg_tensor = paddle.stack(neg_samples, axis=0)
        return neg_tensor

    def _evaluate(self, o_emb, d_ck, d_rg, r, rel_embs, poi_embs, neg_sample, subgraph_alias):
        # o_emb_dup: b x l x h
        # d_ck: b x l
        if isinstance(neg_sample, tuple): # coarse
            neg_sample_poi, neg_sample_region = neg_sample
            eval_sample = neg_sample_poi
            eval_sample_region = neg_sample_region
            eval_sample = eval_sample.reshape([1, -1]).expand([d_ck.shape[0], -1])
            eval_sample_region = eval_sample_region.reshape([1, -1]).expand([d_rg.shape[0], -1])
            eval_sample_re = paddle.stack([subgraph_alias[i] for i in eval_sample], axis=0)

            eval_sample_emb = paddle.stack([poi_embs[i] for i in eval_sample_re], axis=0) # b x l x h
            eval_sample_emb = self._relation(eval_sample_emb, r, rel_embs, self.args.trans)
            o_emb_dup = o_emb.unsqueeze(1).expand_as(eval_sample_emb) # b x l x h
            score = paddle.norm(o_emb_dup - eval_sample_emb, p=2, axis=-1) # b x l x h
            return score, eval_sample, eval_sample_region
        else: # fine
            neg_sample_poi = neg_sample
            eval_sample = paddle.concat([d_ck, neg_sample_poi], axis=-1)
            eval_sample_re = paddle.stack([subgraph_alias[i] for i in eval_sample], axis=0)

            eval_sample_emb = paddle.stack([poi_embs[i] for i in eval_sample_re], axis=0) # b x l x h
            eval_sample_emb = self._relation(eval_sample_emb, r, rel_embs, self.args.trans)
            o_emb_dup = o_emb.unsqueeze(1).expand_as(eval_sample_emb) # b x l x h
            score = paddle.norm(o_emb_dup - eval_sample_emb, p=2, axis=-1) # b x l x h
            score = paddle.where(eval_sample == 0, paddle.full(shape=score.shape, fill_value=np.inf), score) # mask
            return score, eval_sample
    
    def _relation(self, emb, r, rel_embs, mode='transe'):
        # o_emb: b x h
        # d_emb: b x l x h
        if mode == 'transd':
            relation = rel_embs[r] # b x h
            if len(emb.shape) == 2:
                trans_mat = paddle.matmul(relation.unsqueeze(2), self.head_linear(emb).unsqueeze(1)) # b x h x h
                emb_r = paddle.bmm(emb.unsqueeze(1), trans_mat).squeeze(1)
                return emb_r
            elif len(emb.shape) == 3:
                trans_mat = paddle.matmul(relation.reshape([relation.shape[0], 1, -1, 1]).expand([-1, emb.shape[1], -1, -1]), self.tail_linear(emb).unsqueeze(2)) # b x h x h
                emb_r = paddle.matmul(emb.unsqueeze(2), trans_mat).squeeze(2)
                return emb_r
        if mode == 'transe':
            return emb
    
    def _alias(self, bids, n_poi):
        alias = paddle.zeros([n_poi], dtype='int64')
        for idx, b in enumerate(bids):
            alias[b] = idx
        return alias
    
    def _union_eval_sample(self, region_poi, o_rg, d_ck, batch_size, device, k=100):
        poi_samples, region_samples = [], []
        for t_p, t_or in zip(d_ck, o_rg):
            s = np.random.choice(np.setdiff1d(list(range(1, self.n_poi)), t_p.cpu().tolist() + list(self.region_poi[t_or.item()])), (1, k))
            poi_samples.append(paddle.to_tensor(s))
        poi_samples = paddle.concat(poi_samples, axis=0)
        return poi_samples

    def forward(self, uid, o_ck, d_ck, o_rg, d_rg):
        d_neg_items = self.neg_sampling(d_ck.shape[1], d_ck, is_mask=True) # b x l
        item_involve = paddle.concat([o_ck.flatten(), d_ck.flatten(), d_neg_items.flatten()])
        item_involve = paddle.unique(item_involve)

        # if self.args.crf: sub_pp_adj = self.pp_adj.subgraph(item_involve)
        if self.args.crf: sub_pp_adj = subgraph(self.pp_adj, item_involve)

        # generic graph
        node_repr = self.poi_embedding(item_involve)
        if self.args.crf:
            poi_repr, crf_loss = self.crf(node_repr, sub_pp_adj)
        else:
            poi_repr = node_repr
        region_repr = self.region_embedding.weight

        # reindex by alias
        subgraph_alias = self._alias(item_involve, self.n_poi)
        o_ck_re = paddle.stack([subgraph_alias[i] for i in o_ck], axis=0)
        d_ck_re = paddle.stack([subgraph_alias[i] for i in d_ck], axis=0)
        d_neg_re = paddle.stack([subgraph_alias[i] for i in d_neg_items], axis=0)

        # home town avg pooling
        # o_items_emb = poi_repr[o_ck_re] #self.poi_embedding(o_ck)
        o_items_emb = paddle.stack([poi_repr[i] for i in o_ck_re], axis=0)
        o_emb = self._avg_pooling(o_ck, o_items_emb) # b x h
        o_emb = self._relation(o_emb, o_rg, region_repr, self.args.trans) # B x H

        # out-of-town avg pooling
        # d_items_emb = poi_repr[d_ck_re] #self.poi_embedding(o_ck)
        d_items_emb = paddle.stack([poi_repr[i] for i in d_ck_re], axis=0)
        d_emb = self._avg_pooling(d_ck, d_items_emb) # b x h
        d_emb = self._relation(d_emb, o_rg, region_repr, self.args.trans) # B x H

        # memory write
        if self.args.memory:
            mem = self.memory.get_memory()
            for u in range(uid.shape[0]):
                mem = self.memory(memory=mem, write_pack=(d_emb[u], d_rg[u]), state="w")

        # d_neg_emb = poi_repr[d_neg_re] #self.poi_embedding(d_neg_items) # b x l x h
        # d_target_emb = poi_repr[d_ck_re] #self.poi_embedding(d_ck) # b x l x h
        d_neg_emb = paddle.stack([poi_repr[i] for i in d_neg_re], axis=0)
        d_target_emb = paddle.stack([poi_repr[i] for i in d_ck_re], axis=0)
        d_neg_emb = self._relation(d_neg_emb, o_rg, region_repr, self.args.trans)
        d_target_emb = self._relation(d_target_emb, o_rg, region_repr, self.args.trans)

        if self.args.memory:
            user_mem_batch = self.memory(memory=mem, read_pack=(o_emb, d_rg), state="ra") # B x N x H
            ada_mem_emb, reg_attn = self.reg_attn(
                o_emb.unsqueeze(1).expand([-1, self.n_region, -1]),
                user_mem_batch,
                user_mem_batch) # B x H
            mem_emb, _ = self.memory(memory=mem, read_pack=(o_emb, d_rg), state='r')
            fusion_emb = self.fusion_mlp(paddle.concat([o_emb, mem_emb], axis=-1))
            # fusion_emb = o_emb + mem_emb
            o_emb_dup = fusion_emb.unsqueeze(1).expand_as(d_target_emb)
            con_loss = paddle.norm(ada_mem_emb - mem_emb, p=2, axis=-1).mean()
        else:
            o_emb_dup = o_emb.unsqueeze(1).expand_as(d_target_emb)

        s_pos = paddle.norm(o_emb_dup - d_target_emb, p=2, axis=-1) ** 2 # b x l
        s_neg = paddle.norm(o_emb_dup - d_neg_emb, p=2, axis=-1) ** 2 # b x l

        loss = F.relu(s_pos - s_neg + self.args.margin).mean()
        if self.args.crf: loss += crf_loss
        if self.args.memory: loss += con_loss
        
        return loss, poi_repr#, score, alias_poi, alias_region
    
    def coarse_rank(self, uid, o_ck, d_ck, o_rg, d_rg, candidate_size=10):
        #eval_p, eval_r = eval_sampling(self.region_poi, d_ck.size(0), self.args.device, k=10)
        #eval_p, eval_r = self.eval_samples
        if self.eval_p is None or self.eval_r is None:
            p_emb_list = []
            del(self.eval_r)
            del(self.eval_p)
            self.eval_r = []
            self.eval_p = []
            print("generating eval list...")
            for r, p in self.region_poi.items():
                p_emb = self.poi_embedding(paddle.to_tensor(list(p)))
                if self.args.crf:
                    sub_pp_adj = subgraph(self.pp_adj, list(p))
                    p_emb, _ = self.crf(p_emb, sub_pp_adj)
                center_emb = paddle.mean(p_emb, axis=0).reshape([1, -1])
                dist = paddle.norm(center_emb - p_emb, axis=-1)
                _, top_alias = dist.topk(k=candidate_size, largest=False)
                top_r = paddle.full_like(top_alias, r)
                top_p_emb = paddle.stack([p_emb[i] for i in top_alias], axis=0)
                top_p = paddle.to_tensor(list(p))
                top_p = paddle.stack([top_p[i] for i in top_alias], axis=0)
                self.eval_p.append(top_p)
                self.eval_r.append(top_r)
                #
                p_emb_list.append(top_p_emb)
            p_emb_list = paddle.concat(p_emb_list, axis=0)
            self.eval_r = paddle.concat(self.eval_r, axis=0)
            self.eval_p = paddle.concat(self.eval_p, axis=0)

        eval_p = self.eval_p
        eval_r = self.eval_r

        item_involve = paddle.concat([o_ck.flatten(), d_ck.flatten(), eval_p.flatten()])
        item_involve = paddle.unique(item_involve)

        if self.args.crf: sub_pp_adj = subgraph(self.pp_adj, item_involve)

        # generic graph
        node_repr = self.poi_embedding(item_involve)
        if self.args.crf: 
            poi_repr, _ = self.crf(node_repr, sub_pp_adj)
        else:
            poi_repr = node_repr
        # poi_repr = node_repr
        region_repr = self.region_embedding.weight

        # reindex by alias
        subgraph_alias = self._alias(item_involve, self.n_poi)
        o_ck_re = paddle.stack([subgraph_alias[i] for i in o_ck], axis=0)
        
        # home town avg pooling
        o_items_emb = paddle.stack([poi_repr[i] for i in o_ck_re], axis=0) #self.poi_embedding(o_ck)
        o_emb = self._avg_pooling(o_ck, o_items_emb) # b x h
        o_emb = self._relation(o_emb, o_rg, region_repr, self.args.trans)

        if self.args.memory:
            mem = self.memory.get_memory()
            user_mem_batch = self.memory(memory=mem, read_pack=(o_emb, d_rg), state="ra") # B x N x H
            ada_mem_emb, reg_attn = self.reg_attn(
                o_emb.unsqueeze(1).expand([-1, self.n_region, -1]),
                user_mem_batch,
                user_mem_batch) # B x H, B x N x 1
            fusion_emb = self.fusion_mlp(paddle.concat([o_emb, ada_mem_emb], axis=-1))
            # fusion_emb = o_emb + ada_mem_emb
        else:  
            fusion_emb = o_emb
        
        #o_emb = F.normalize(o_emb, 2, -1)
        score, alias_poi, alias_region = self._evaluate(fusion_emb, d_ck, d_rg, o_rg, region_repr, poi_repr, (eval_p, eval_r), subgraph_alias) # b x (l + k) to-do
        
        return score, alias_poi, alias_region
    
    def fine_rank(self, uid, o_ck, d_ck, o_rg, d_rg, k=200):
        # region_mask = self.region_mask[d_rg]
        # region_poi = self.region_poi[d_rg]
        # prob_dist = paddle.distribution.Categorical(region_mask)
        # samples = prob_dist.sample(shape=[k]).transpose(0, 1) # B x k
        samples = paddle.to_tensor([random.sample(list(self.region_poi[i] - set(d_ck[idx].tolist())), k) for idx, i in enumerate(d_rg.numpy().flatten())])

        item_involve = paddle.concat([o_ck.flatten(), d_ck.flatten(), samples.flatten()])
        item_involve = paddle.unique(item_involve)

        if self.args.crf: sub_pp_adj = subgraph(self.pp_adj, item_involve)

        # generic graph
        node_repr = self.poi_embedding(item_involve)
        if self.args.crf: 
            poi_repr, _ = self.crf(node_repr, sub_pp_adj)
        else:
            poi_repr = node_repr
        region_repr = self.region_embedding.weight

        # reindex by alias
        subgraph_alias = self._alias(item_involve, self.n_poi)
        o_ck_re = paddle.stack([subgraph_alias[i] for i in o_ck], axis=0)

        o_items_emb = paddle.stack([poi_repr[i] for i in o_ck_re], axis=0) #self.poi_embedding(o_ck)
        o_emb = self._avg_pooling(o_ck, o_items_emb) # b x h
        o_emb = self._relation(o_emb, o_rg, region_repr, self.args.trans)

        if self.args.memory:
            mem = self.memory.get_memory()
            mem_emb, slot_attn_weight = self.memory(memory=mem, read_pack=(o_emb, d_rg), state='r')
            fusion_emb = self.fusion_mlp(paddle.concat([o_emb, mem_emb], axis=-1))
            # fusion_emb = o_emb + mem_emb
        else:
            fusion_emb = o_emb
        
        #o_emb = F.normalize(o_emb, 2, -1)
        score, alias_poi = self._evaluate(fusion_emb, d_ck, d_rg, o_rg, region_repr, poi_repr, samples, subgraph_alias) # b x (l + k) to-do

        return score, alias_poi
    
    def union_rank(self, uid, o_ck, d_ck, o_rg, d_rg):
        eval_p = self._union_eval_sample(self.region_poi, o_rg, d_ck, uid.shape[0], self.args.device)
        item_involve = paddle.concat([o_ck.flatten(), d_ck.flatten(), eval_p.flatten()])
        item_involve = paddle.unique(item_involve)

        if self.args.crf: sub_pp_adj = subgraph(self.pp_adj, item_involve)

        # generic graph
        node_repr = self.poi_embedding(item_involve)
        if self.args.crf: 
            poi_repr, _ = self.crf(node_repr, sub_pp_adj)
        else:
            poi_repr = node_repr
        region_repr = self.region_embedding.weight

        # reindex by alias
        subgraph_alias = self._alias(item_involve, self.n_poi)
        o_ck_re = paddle.stack([subgraph_alias[i] for i in o_ck], axis=0)
        
        # home town avg pooling
        o_items_emb = paddle.stack([poi_repr[i] for i in o_ck_re], axis=0) #self.poi_embedding(o_ck)
        o_emb = self._avg_pooling(o_ck, o_items_emb) # b x h
        o_emb = self._relation(o_emb, o_rg, region_repr, self.args.trans)

        if self.args.memory:
            mem = self.memory.get_memory()
            user_mem_batch = self.memory(memory=mem, read_pack=(o_emb, d_rg), state="ra") # B x N x H
            ada_mem_emb, reg_attn = self.reg_attn(
                o_emb.unsqueeze(1).expand([-1, self.n_region, -1]),
                user_mem_batch,
                user_mem_batch) # B x H, B x N x 1
            fusion_emb = self.fusion_mlp(paddle.concat([o_emb, ada_mem_emb], axis=-1))
        else:  
            fusion_emb = o_emb

        score, alias_poi = self._evaluate(fusion_emb, d_ck, d_rg, o_rg, region_repr, poi_repr, eval_p, subgraph_alias)
        return score, alias_poi
