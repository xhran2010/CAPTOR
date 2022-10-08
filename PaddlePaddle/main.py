# -*- coding: utf-8 -*-
import paddle
from paddle.io import DataLoader

import argparse
from collections import namedtuple, defaultdict
import numpy as np
import os
import sys
from copy import copy
import warnings
warnings.filterwarnings('ignore')

try:
    import ipdb
except:
    pass

from utils import *
from data import TravelDataset, random_split
from core import Dyna as CRFDyna

import metrics
from trainer import *

import pickle

def main():
    parser = argparse.ArgumentParser()
    # dataset arguments
    parser.add_argument('--ori_data', type=str, default='../dataset/home.txt')
    parser.add_argument('--dst_data', type=str, default='../dataset/oot.txt')
    parser.add_argument('--trans_data', type=str, default='../dataset/travel.txt')
    parser.add_argument('--feat_path', type=str, default='./feat')
    parser.add_argument('--save_path', type=str, default='./model_save')
    parser.add_argument("--best_save", action="store_true")
    parser.add_argument("--pp_graph_path", type=str, default="../dataset/pp_adj_1.npz")
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--data_split_path', type=str, default='./data_split/us.npy')
    # training configurations
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--margin', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--lr_dc', type=float, default=0.2)
    parser.add_argument('--lr_dc_step', type=int, default=20)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=9875)
    parser.add_argument('--log_path', type=str, default='./log/')
    parser.add_argument('--log', action="store_true")
    parser.add_argument('--name', type=str, default="default")
    parser.add_argument('--model', type=str, default="base")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--trans', type=str, default="transe")
    parser.add_argument("--stop_epoch", type=int, default=8) # early stopping
    parser.add_argument("--fine_stop", type=int, default=12)
    parser.add_argument("--k", type=int, default=50) # k in kNN
    # crf arguments
    parser.add_argument("--crf", action="store_true")
    parser.add_argument("--crf_layer", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    # Memory arguments
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--mem_slot", type=int, default=5)

    args = parser.parse_args()
    set_seeds(args.seed)
    args.save_path = os.path.join(args.save_path, args.name)
    path_exist(args.save_path)

    logger = Logger(args.log_path, args.name, args.seed, args.log)
    logger.log(str(args))
    logger.log("Experiment name: %s" % args.name)

    paddle.set_device('gpu')

    data = TravelDataset(args, args.ori_data, args.dst_data, args.trans_data, args.feat_path)
    train_data, valid_data, test_data = random_split(data, split_path=args.data_split_path)

    # train_loader = DataLoader(train_data, args.train_batch, shuffle=True, collate_fn=collate_fn)
    train_loader = DataLoader(dataset=train_data, batch_size=args.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.test_batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch, shuffle=False, collate_fn=collate_fn)

    n_region = len(data.region_idx)
    r_mask = data.region_mask()
    r_prop = data.get_proportion()
    pp_adj = data.pp_adj

    model = CRFDyna(args, len(data.poi_idx) + 1, len(data.region_idx), data.eval_samples, 
        data.region_poi, r_mask, pp_adj)

    if args.mode == 'train':
        union_best, refine_best = train_single_phase(model, train_loader, valid_loader, args, logger, n_region, r_prop)

        test(model, os.path.join(args.save_path, "model_{}.xhr".format(union_best)), test_loader, args, logger, n_region, r_prop, test_type='union')
        test(model, os.path.join(args.save_path, "model_{}.xhr".format(refine_best)), test_loader, args, logger, n_region, r_prop, test_type='refine')
        print("################## current exp done ##################")
    elif args.mode == 'test':
        test(model, args.test_path, test_loader, args, logger, n_region, r_prop, test_type='refine')

    logger.close_log()
    
if __name__ == "__main__":
    main()