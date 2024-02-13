from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json, logging
import os, sys
from statistics import mode

import random
from tkinter import E
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import optimizer

import time
from model.model import *
from data_process import get_dataset

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.linalg import sqrtm
from scipy.stats import norm
from torch_geometric.utils import negative_sampling as pyg_negative_sampling


from utils import *
from model.base_gnn import *
from model.model_cond import *


torch.set_printoptions(profile='full')
np.set_printoptions(precision=4, threshold=sys.maxsize)


def mean_trials(out_list, name='', log_file=None):
    ci = 1.96 * st.sem(out_list) / math.sqrt(len(out_list))
    log = name + ' Mean: {:.3f} Std: {:.3f}' \
            .format(np.mean(out_list), st.sem(out_list)) 
    print(log)
    return log



def evaluate(model, idx, mode='test'):

    neg_num=9
    labels = np.zeros(neg_num+1)
    labels[0] = 1

    pos_score = np.empty([0])
    neg_score = np.empty([0,9])
    num_sample = idx.shape[0]

    model.eval()
    with torch.no_grad():
        output = model(data.feat, data.adj)

    a = output[idx[:,0]]            
    b = output[idx[:,1]]
    c = output[idx[:,2:11]]

    pos = torch.sigmoid(torch.sum(torch.mul(a,b), dim=1))
    neg = torch.sigmoid(torch.sum(torch.mul(a.view(a.shape[0],1,a.shape[1]),c), dim=2))

    pos_score = np.concatenate((pos_score, pos.cpu().detach().numpy()))
    neg_score = np.concatenate((neg_score, neg.cpu().detach().numpy()))

    pred_list = np.concatenate((np.expand_dims(pos_score, axis=1), neg_score), axis=1)

    sum_ndcg = 0
    sum_mrr = 0
    sum_hit1 = 0

    for i in range(num_sample):
        # in our setting, MAP = MRR, use MRR for faster computation                
        true = pred_list[i, 0]
        sort_list = np.sort(pred_list[i])[::-1]


        rank = int(np.where(sort_list == true)[0][0]) + 1
        sum_mrr += (1/rank)

        if mode == 'test':
            if pred_list[i, 0] == np.max(pred_list[i]):
                sum_hit1 += 1

            NDCG = ndcg_score([labels], [pred_list[i]])
            sum_ndcg += NDCG
            # AP = average_precision_score(labels, pred_list[i])
            # sum_map += AP

    H1 = sum_hit1 / num_sample
    MRR = sum_mrr / num_sample
    NDCG = sum_ndcg / num_sample

    if mode == 'test':
        log = "MAP/MRR={:.3f}, NDCG={:.3f}, H1={:.3f}" \
            .format(MRR, NDCG , H1) 
        print(log)
        return(MRR, NDCG , H1)
    else:
        log = "MAP/MRR={:.3f}".format(MRR) 
        print(log)
        return (MRR, 0, 0)


def run(args, data, save_path, seed, device):

    model_path = os.path.join(save_path, 'model_{:d}.pkl'.format(seed))
    in_feat = data.feat.shape[1]

    model = GCN(in_feat, args.nhid, args)
    # model = GraphSAGE(in_feat, args.nhid, args)
    # model = GAT(in_feat, args.nhid, args)

    optimizer = torch.optim.Adam(model.parameters(),\
                                 lr=args.lr, weight_decay=args.decay)

    diffusion = Diffusion_Cond(args.nhid, args.nhid, args, args.nhid)
    d_optimizer = torch.optim.Adam(diffusion.parameters(),\
                                 lr=args.lr, weight_decay=args.decay)

    n_nodes = data.feat.shape[0]
    num = 20
    
    # if args.cuda:
    data.to_device(device) #.to_cuda()
    model = model.to(device)
    diffusion = diffusion.to(device)

    no_self = data.adj - torch.eye(data.num_ent, device=device)
    
    neigh = torch.zeros(num*n_nodes, dtype=int, device=device)

    for i in range(n_nodes): 
        nei = torch.where(no_self[i] > 0)[0]
        nei = nei[torch.randperm(nei.shape[0])]
        size = nei.shape[0]
        if size > num:
            nei1 = nei[num]
        elif size == 0:
            nei1 = i
        else:
            nei1 = torch.zeros(num, device=device)
            for j in range(num):
                nei1[j] = nei[j%size]
        neigh[i*num:i*num+num] = nei1
     
    if not args.no_train:
        best = 0
        cur = 0

        print('Training ...')
        for i in range(args.epoch):
            model.train()
            optimizer.zero_grad()

            output = model(data.feat, data.adj)

            idx = data.t['train']
            a = output[idx[:,0]]            
            b = output[idx[:,1]]
            c = output[idx[:,2]]
           
            #op = nn.LogSigmoid()
            pos = torch.sum(a*b, dim=1)
            pos = -torch.mean(F.sigmoid(pos))  #F.logsigmoid(pos)) 
            
            neg = torch.sum(a*c, dim=1)
            neg = -torch.mean(F.sigmoid(-neg)) #F.logsigmoid(-neg)

    
            def train_diffusion():            
                nei_output = output[neigh].detach()
                n_output = output[np.repeat(np.arange(n_nodes, dtype=int), num)].detach()

                for epoch in range(args.d_epoch):
                    d_optimizer.zero_grad()
                    
                    dif_loss = diffusion(nei_output, n_output, device)
                    dif_loss.backward(retain_graph=True)
                    d_optimizer.step()
                    #print("Diffusion_Loss:", dif_loss.item())
                return


            if not args.no_diff:
                train_diffusion()
                
                h_syn = diffusion.sample(a.shape, a)
                neg_list = []
                w = [0, 1, 0.9, 0.8, 0.7]
                
                for i in range(len(h_syn)):
                    syn_neg = torch.sum(a*h_syn[i], dim=1)
                    syn_neg = -torch.mean(F.sigmoid(-syn_neg))  #F.logsigmoid(-syn_neg)
                    neg_list.append(w[i] * syn_neg)
                    
                neg_list.append(neg)
                sam = [1 for i in w if i != 0]
                neg = sum(neg_list) / (sum(sam)+1)

            loss = (pos + neg)/2 
            loss.backward()
            optimizer.step()

            val_mrr, _, _ = evaluate(model, data.t['val'], 'val')

            if best <= val_mrr:
                best = val_mrr
                cur = 0
                torch.save({'model': model,
                            'diffusion': diffusion}, model_path)
            else:
                cur += 1
            
            if cur > args.patience:
                print('Early stopping!')
                break
    
    # Test
    print('Testing ...')

    models = torch.load(model_path)
    model = models['model']
    diffusion = models['diffusion']

    map, ndcg, h1 = evaluate(model, data.t['test'], 'test')
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        f.write('MAP: '.join([str(map)]) + '\n')
        f.write('NDCG: '.join([str(ndcg)]) + '\n')
        f.write('H1: '.join([str(h1)]) + '\n')

    return map, ndcg, h1
    
    

if __name__ == '__main__':
    
    args = parse_args()
    
    # device = "cuda:"+str(args.gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device("mps")

    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

    np.random.seed(args.seed)
    seed = np.random.choice(100, args.n_runs, replace=False)
    print('Seed: ', seed)

    ndcg_list = []
    mrr_list = []
    h1_list = []

    print('Processing data ...')        
    data = get_dataset(args, args.dataset)
    save_path = prepare_saved_path(args)

    for i in range(args.n_runs): 
        np.random.seed(seed[i])
        torch.manual_seed(seed[i])
        # if args.cuda:
        #     torch.cuda.set_device(args.gpu)
        #     torch.cuda.manual_seed(seed[i])
        mrr, ndcg, h1 = run(args, data, save_path, seed[i], device)

        mrr_list.append(mrr)
        ndcg_list.append(ndcg)
        h1_list.append(h1)

    mean_trials(mrr_list, name='MAP')
    mean_trials(ndcg_list, name='NDCG')
    mean_trials(h1_list, name='H1')