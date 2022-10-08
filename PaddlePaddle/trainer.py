# -*- coding: utf-8 -*-

from paddle.optimizer import Adam, lr

import argparse
from collections import namedtuple, defaultdict, Counter
import numpy as np
import os
import sys

from copy import copy

from utils import *
import metrics

import pickle

try:
    from tqdm import tqdm
    import ipdb
except:
    pass

def train_single_phase(model, train_loader, valid_loader, args, logger, n_region, r_prop):
    scheduler = lr.StepDecay(learning_rate=args.lr, step_size=args.lr_dc_step, gamma=args.lr_dc)
    optimizer = Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=args.l2)

    stopping_dict = defaultdict(float)
    union_flag = True
    refine_flag = True

    for e in range(args.epoch):
        # train
        model.train() # train mode
        model.eval_p = None
        model.eval_r = None
        loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
        for b, (uid, o_ck, d_ck, o_rg, d_rg) in tqdm(enumerate(train_loader), total=len(train_loader.dataset) / args.train_batch):
            loss, crf_output = model(uid, o_ck, d_ck, o_rg, d_rg)
            #logger.log("rec loss: {:.8f}, crf loss: {:.8f}".format(rec_loss.item(), crf_loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_sum += loss.item()
        scheduler.step()
        logger.log("Epoch %d/%d : Train Loss %.10f" % (e, args.epoch - 1, loss_sum / (b + 1)))
        if e % args.save_step == 0 and not args.best_save:
            save_model(model, e, args.save_path)
        # validation
        model.eval()

        #################### union ####################
        if union_flag:
            poi_rec_10 = 0.
            poi_precision_10 = 0.
            poi_f1_10 = 0.
            poi_ndcg_10 = 0.

            for b, (uid, o_ck, d_ck, o_rg, d_rg) in tqdm(enumerate(valid_loader), total=len(valid_loader.dataset)/args.test_batch):                
                score, alias_poi = model.union_rank(uid, o_ck, d_ck, o_rg, d_rg)
                _, alias_top = score.topk(k=30,axis=1,largest=False)
                real_top = paddle.stack([paddle.gather(x, idx) for x, idx in zip(alias_poi, alias_top)], axis=0)
                real_top = real_top.cpu().detach().numpy() # B x k
                label = d_ck.cpu().detach().numpy()

                poi_rec_10 += metrics.p_rec(real_top, label, k=10)
                poi_precision_10 += metrics.p_precision(real_top, label, k=10)
                poi_f1_10 += metrics.p_f1(real_top, label, k=10)
                poi_ndcg_10 += metrics.p_ndcg(real_top, label, k=10)

            # independent poi-level metrics
            poi_rec_10 /= len(valid_loader.dataset)
            poi_precision_10 /= len(valid_loader.dataset)
            poi_f1_10 /= len(valid_loader.dataset)
            poi_ndcg_10 /= len(valid_loader.dataset)

            logger.log("[val-union] Epoch {}/{} P-Rec@10: {:5.4f} P-Precision@10: {:5.4f} P-F1@10: {:5.4f} P-NDCG@10: {:5.4f}" \
            .format(e, args.epoch - 1, poi_rec_10, poi_precision_10, poi_f1_10, poi_ndcg_10))

        if refine_flag:
            #################### coarse ####################
            region_acc = 0.

            fine_users = defaultdict(list)
            n_fine = 0

            for b, (uid, o_ck, d_ck, o_rg, d_rg) in tqdm(enumerate(valid_loader), total=len(valid_loader.dataset)/args.test_batch):                
                score, alias_poi, alias_region = model.coarse_rank(uid, o_ck, d_ck, o_rg, d_rg)
                
                _, alias_top = score.topk(k=args.k,axis=1,largest=False)
                region_top = paddle.stack([paddle.gather(x, idx) for x, idx in zip(alias_region, alias_top)], axis=0)

                region_predict = np.argmax(np.stack([np.bincount(i, minlength=n_region) for i in region_top.numpy()]), axis=-1)

                real_top = paddle.stack([paddle.gather(x, idx) for x, idx in zip(alias_poi, alias_top)], axis=0)
                real_top = real_top.cpu().detach().numpy() # B x k
                label = d_ck.cpu().detach().numpy()

                region_acc += metrics.r_acc(region_predict, d_rg)

                for u, t in enumerate(region_predict == d_rg.numpy()):
                    if t:
                        fine_users[d_rg[u].item()].append((uid[u], o_ck[u], d_ck[u], o_rg[u], d_rg[u]))
                        n_fine += 1
                #print("{:5.4f} {:5.4f}".format(metrics.r_precision(region_top, d_rg), metrics.r_precision(region_top, d_rg, batch_w)))
            
            region_acc /= (b + 1)

            logger.log("[val-region] Epoch {}/{} R-Acc: {:5.4f}" \
            .format(e, args.epoch - 1, region_acc))

            if n_fine == 0:
                logger.log("[val-poi] Epoch {}/{} no user for fine ranking!".format(e, args.epoch - 1))
                continue

            # fine rank
            #################### fine ####################
            step_len = 8

            fine_poi_rec_10 = 0.
            fine_poi_precision_10 = 0.
            fine_poi_f1_10 = 0.
            fine_poi_ndcg_10 = 0.

            for k_, v_ in fine_users.items():
                for i in range(0, len(v_), step_len):
                    uid, o_ck, d_ck, o_rg, d_rg = collate_fn(v_[i:i + step_len])

                    score, alias_poi = model.fine_rank(uid, o_ck, d_ck, o_rg, d_rg)
                    _, alias_top = score.topk(k=30,axis=1,largest=False)
                    real_top = paddle.stack([paddle.gather(x, idx) for x, idx in zip(alias_poi, alias_top)], axis=0)
                    real_top = real_top.cpu().detach().numpy() # B x k
                    label = d_ck.cpu().detach().numpy()

                    fine_poi_rec_10 += metrics.p_rec(real_top, label, k=10)
                    fine_poi_precision_10 += metrics.p_precision(real_top, label, k=10)
                    fine_poi_f1_10 += metrics.p_f1(real_top, label, k=10)
                    fine_poi_ndcg_10 += metrics.p_ndcg(real_top, label, k=10)
            
            # independent poi-level metrics
            fine_poi_rec_10 /= len(valid_loader.dataset)
            fine_poi_precision_10 /= len(valid_loader.dataset)
            fine_poi_f1_10 /= len(valid_loader.dataset)
            fine_poi_ndcg_10 /= len(valid_loader.dataset)

            logger.log("[val-poi] Epoch {}/{} #fine users: {} P-Rec@10: {:5.4f} P-Precision@10: {:5.4f} P-F1@10: {:5.4f} P-NDCG@10: {:5.4f}" \
            .format(e, args.epoch - 1, n_fine, fine_poi_rec_10, fine_poi_precision_10, fine_poi_f1_10, fine_poi_ndcg_10))

        # union early stop
        if union_flag:
            if poi_f1_10 > stopping_dict['best_f1']:
                stopping_dict['best_f1'] = poi_f1_10
                stopping_dict['f1_epoch'] = 0
                stopping_dict['best_union_epoch'] = e
                if args.best_save: save_model(model, "union_best", args.save_path, optimizer, scheduler)
            else:
                stopping_dict['f1_epoch'] += 1

            if poi_ndcg_10 > stopping_dict['best_ndcg']:
                stopping_dict['best_ndcg'] = poi_ndcg_10
                stopping_dict['ndcg_epoch'] = 0
            else:
                stopping_dict['ndcg_epoch'] += 1

            logger.log("union early stop: {}|{}".format(stopping_dict['f1_epoch'], stopping_dict["ndcg_epoch"]))

            if stopping_dict['f1_epoch'] >= args.stop_epoch or stopping_dict['ndcg_epoch'] >= args.stop_epoch:
                union_flag = False
                logger.log("union early stopped! best epoch: {}".format(stopping_dict['best_union_epoch']))

                union_return = stopping_dict['best_union_epoch']
        
        # refine early stop
        if refine_flag:
            if region_acc > stopping_dict['best_racc']:
                stopping_dict['best_racc'] = region_acc
                stopping_dict['racc_epoch'] = 0
                stopping_dict['best_refine_epoch'] = e
                if args.best_save: save_model(model, "refine_best", args.save_path, optimizer, scheduler)
            else:
                stopping_dict['racc_epoch'] += 1

            logger.log("refine early stop: {}".format(stopping_dict['racc_epoch']))

            if stopping_dict['racc_epoch'] >= args.fine_stop:
                refine_flag = False
                logger.log("refine early stopped! best epoch: {}".format(stopping_dict['best_refine_epoch']))

                refine_return = stopping_dict['best_refine_epoch']
        
        if not union_flag and not refine_flag:
            if args.best_save:
                return "union_best", "refine_best"
            else:
                return union_return, refine_return

def test(model, model_path, test_loader, args, logger, n_region, r_prop, test_type="union"):
    checkpoint = paddle.load(model_path) 
    model.set_state_dict(checkpoint)
    model.eval()
    model.eval_p = None
    model.eval_r = None

    if test_type == 'union':
        ###################### union ######################
        poi_rec_5 = 0.
        poi_precision_5 = 0.
        poi_ndcg_5 = 0.

        poi_rec_10 = 0.
        poi_precision_10 = 0.
        poi_ndcg_10 = 0.

        poi_rec_15 = 0.
        poi_precision_15 = 0.
        poi_ndcg_15 = 0.

        for b, (uid, o_ck, d_ck, o_rg, d_rg) in tqdm(enumerate(test_loader), total=len(test_loader.dataset) / args.test_batch):
            score, alias_poi = model.union_rank(uid, o_ck, d_ck, o_rg, d_rg)
            alias_score, alias_top = score.topk(k=100,axis=1,largest=False)
            real_top = paddle.stack([paddle.gather(x, idx) for x, idx in zip(alias_poi, alias_top)], axis=0)
            real_top = real_top.cpu().detach().numpy() # B x k

            label = d_ck.cpu().detach().numpy()

            poi_rec_5 += metrics.p_rec(real_top, label, k=5)
            poi_precision_5 += metrics.p_precision(real_top, label, k=5)
            poi_ndcg_5 += metrics.p_ndcg(real_top, label, k=5)

            poi_rec_10 += metrics.p_rec(real_top, label, k=10)
            poi_precision_10 += metrics.p_precision(real_top, label, k=10)
            poi_ndcg_10 += metrics.p_ndcg(real_top, label, k=10)

            poi_rec_15 += metrics.p_rec(real_top, label, k=15)
            poi_precision_15 += metrics.p_precision(real_top, label, k=15)
            poi_ndcg_15 += metrics.p_ndcg(real_top, label, k=15)

        poi_rec_5 /= len(test_loader.dataset)
        poi_precision_5 /= len(test_loader.dataset)
        poi_ndcg_5 /= len(test_loader.dataset)

        poi_rec_10 /= len(test_loader.dataset)
        poi_precision_10 /= len(test_loader.dataset)
        poi_ndcg_10 /= len(test_loader.dataset)

        poi_rec_15 /= len(test_loader.dataset)
        poi_precision_15 /= len(test_loader.dataset)
        poi_ndcg_15 /= len(test_loader.dataset)

        logger.log("[test-general] HR@5: {:5.4f} Pre@5: {:5.4f} NDCG@5: {:5.4f}, HR@10: {:5.4f} Pre@10: {:5.4f} NDCG@10: {:5.4f}, HR@15: {:5.4f} Pre@15: {:5.4f} NDCG@15: {:5.4f}, " \
        .format(poi_rec_5, poi_precision_5, poi_ndcg_5, poi_rec_10, poi_precision_10, poi_ndcg_10, poi_rec_15, poi_precision_15, poi_ndcg_15))

    elif test_type == "refine":
        ###################### coarse ######################
        fine_users = defaultdict(list)
        fine_user_list = []
        real_top_list = []
        real_top_score = []
        r_map_score = []

        region_acc = 0.
        # coarse rank
        n_fine = 0

        for b, (uid, o_ck, d_ck, o_rg, d_rg) in tqdm(enumerate(test_loader), total=len(test_loader.dataset) / args.test_batch):

            score, alias_poi, alias_region = model.coarse_rank(uid, o_ck, d_ck, o_rg, d_rg, candidate_size=10)
            n_region = model.n_region

            alias_score, alias_top = score.topk(k=10,axis=1,largest=False)

            region_top = paddle.stack([paddle.gather(x, idx) for x, idx in zip(alias_region, alias_top)], axis=0)

            region_predict = np.argmax(np.stack([np.bincount(i, minlength=n_region) for i in region_top.numpy()]), axis=-1)

            # side output
            for u, t in enumerate(region_predict == d_rg.numpy()):
                if t: 
                    fine_users[d_rg[u].item()].append((uid[u], o_ck[u], d_ck[u], o_rg[u], d_rg[u]))
                    n_fine += 1

            region_acc += metrics.r_acc(region_predict, d_rg)

        region_acc /= (b + 1)

        logger.log("[test-region] R-Acc: {:5.4f}".format(region_acc))
        ###################### fine ######################

        #return
        # fine rank
        step_len = 8

        poi_rec_5 = 0.
        poi_precision_5 = 0.
        poi_ndcg_5 = 0.

        poi_rec_10 = 0.
        poi_precision_10 = 0.
        poi_ndcg_10 = 0.

        poi_rec_15 = 0.
        poi_precision_15 = 0.
        poi_ndcg_15 = 0.

        for k_, v_ in tqdm(fine_users.items()):
            for i in range(0, len(v_), step_len):
                uid, o_ck, d_ck, o_rg, d_rg = collate_fn(v_[i:i + step_len])
                
                score, alias_poi = model.fine_rank(uid, o_ck, d_ck, o_rg, d_rg)
                alias_score, alias_top = score.topk(k=50,axis=1,largest=False)
                real_top = paddle.stack([paddle.gather(x, idx) for x, idx in zip(alias_poi, alias_top)], axis=0)
                real_top = real_top.cpu().detach().numpy() # B x k
                label = d_ck.cpu().detach().numpy()

                poi_rec_5 += metrics.p_rec(real_top, label, k=5)
                poi_precision_5 += metrics.p_precision(real_top, label, k=5)
                poi_ndcg_5 += metrics.p_ndcg(real_top, label, k=5)

                poi_rec_10 += metrics.p_rec(real_top, label, k=10)
                poi_precision_10 += metrics.p_precision(real_top, label, k=10)
                poi_ndcg_10 += metrics.p_ndcg(real_top, label, k=10)

                poi_rec_15 += metrics.p_rec(real_top, label, k=15)
                poi_precision_15 += metrics.p_precision(real_top, label, k=15)
                poi_ndcg_15 += metrics.p_ndcg(real_top, label, k=15)

        poi_rec_5 /= len(test_loader.dataset)
        poi_precision_5 /= len(test_loader.dataset)
        poi_ndcg_5 /= len(test_loader.dataset)

        poi_rec_10 /= len(test_loader.dataset)
        poi_precision_10 /= len(test_loader.dataset)
        poi_ndcg_10 /= len(test_loader.dataset)

        poi_rec_15 /= len(test_loader.dataset)
        poi_precision_15 /= len(test_loader.dataset)
        poi_ndcg_15 /= len(test_loader.dataset)

        logger.log("[test-poi] HR@5: {:5.4f} Pre@5: {:5.4f} NDCG@5: {:5.4f}, HR@10: {:5.4f} Pre@10: {:5.4f} NDCG@10: {:5.4f}, HR@15: {:5.4f} Pre@15: {:5.4f} NDCG@15: {:5.4f}, " \
        .format(poi_rec_5, poi_precision_5, poi_ndcg_5, poi_rec_10, poi_precision_10, poi_ndcg_10, poi_rec_15, poi_precision_15, poi_ndcg_15))