import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.optim as optim
from sklearn import metrics
import random
import os
import sys
import utils
from tqdm import tqdm
from collections import defaultdict
import time
import argparse
import dgl
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import multiprocessing
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix

from preprocess.data_load import gen_DGLGraph, gen_weighted_DGLGraph
import preprocess.data_load as dl
from preprocess.batch import DataLoader
from initialize.initial_embedder import MultipleEmbedding
from initialize.random_walk_hyper import random_walk_hyper

from model.Whatsnet import Whatsnet, WhatsnetLayer
from model.WhatsnetIM import WhatsnetIM
from model.WhatsnetLSPE import WhatsnetLSPE, WhatsnetLSPELayer
from model.WhatsnetHAT import WhatsnetHAT, WhatsnetHATLayer
from model.WhatsnetHNHN import WhatsnetHNHN, WhatsnetHNHNLayer
from model.layer import FC, Wrap_Embedding

def run_epoch(args, data, dataloader, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer, optim, scheduler, loss_fn, opt="train"):
    total_pred = []
    total_label = []
    num_data = 0
    num_recon_data = 0
    total_loss = 0
    total_ce_loss = 0
    total_recon_loss = 0
    
    # Batch ==============================================================
    ts = time.time()
    batchcount = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (input_nodes, output_nodes, blocks) in pbar:      
        # Wrap up loader
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices = srcs.to(device)
        hedgeindices = dsts.to(device)
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)

        batchcount += 1
        # Get Embedding
        if args.embedder == "whatsnetLSPE":
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            v_pos = data.v_pos[input_nodes['node']].to(device)
            e_pos = data.e_pos[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat, v_pos, e_pos)
        elif args.embedder == "whatsnet":
            if args.att_type_v in ["ITRE", "ShawRE"]:
                vindex = torch.arange(len(input_nodes['node'])).unsqueeze(1).to(device)
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat, vindex)
            else:
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                # concat feature embedding
                color_feat, _ = colorembedder(data.node2color[input_nodes['node']].long().to(device))
                size_feat, _ = sizeembedder(data.node2size[input_nodes['node']].long().to(device))
                group_feat, _ = groupembedder(data.node2group[input_nodes['node']].long().to(device))
                v_feat = torch.concat((v_feat, color_feat, size_feat, group_feat), dim=1)
                if not args.custom:
                    v, e = embedder(blocks, v_feat, e_feat, None, None)
                else:
                    v, e = embedder(blocks, v_feat, e_feat, data, customerembedder)
        else:
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat)
                
        # Predict Class
        if args.scorer == "sm":
          if args.dataset_name == 'task1':
            unique_hedgeindices = torch.unique(hedgeindices)
            label_indices = []
            for i in range(len(hedgeindices)):
                if i == 0:
                  label_indices.append(i)
                  prev_value = hedgeindices[i]
                else:
                  if hedgeindices[i] != prev_value:
                      label_indices.append(i)
                      prev_value = hedgeindices[i]

            label_indices = torch.Tensor(label_indices).long()
            nodelabels = nodelabels[label_indices]   
            hembedding = e[unique_hedgeindices]
            input_embeddings = hembedding
            predictions = scorer(input_embeddings)

          elif args.dataset_name == 'task2':
            if opt == 'train':
              trainable_indices = torch.nonzero((nodelabels == 0) | (nodelabels == 1)).squeeze()
              hedgeindices = hedgeindices[trainable_indices]
              nodeindices = nodeindices[trainable_indices]
              nodelabels = nodelabels[trainable_indices]

              hembedding = e[hedgeindices]
              vembedding = v[nodeindices]
              input_embeddings = torch.cat([hembedding,vembedding], dim=1)
              predictions = scorer(input_embeddings)
            elif opt == 'valid':
              valid_indices = torch.nonzero((nodelabels == 2) | (nodelabels == 3)).squeeze()
              hedgeindices = hedgeindices[valid_indices]
              nodeindices = nodeindices[valid_indices]
              nodelabels = nodelabels[valid_indices]
              nodelabels = nodelabels - 2


              hembedding = e[hedgeindices]
              vembedding = v[nodeindices]
              input_embeddings = torch.cat([hembedding,vembedding], dim=1)
              predictions = scorer(input_embeddings)
              
          else:
            raise ValueError("Not supported data type")
              
        elif args.scorer == "im":
            predictions, nodelabels = scorer(blocks[-1], v, e)
        total_pred.append(predictions.detach())
        total_label.append(nodelabels.detach())
        
        # Back Propagation
        num_data += predictions.shape[0]
        num_recon_data += input_nodes['node'].shape[0]
        ce_loss = loss_fn(predictions, nodelabels)
        loss = ce_loss + args.rw * recon_loss
        if opt == "train":
            optim.zero_grad()
            loss.backward() 
            optim.step()
        total_loss += (loss.item() * predictions.shape[0])
        total_ce_loss += (ce_loss.item() * predictions.shape[0])
        total_recon_loss += (recon_loss.item() * input_nodes['node'].shape[0]) # this is fixed as zero
        if opt == "train":
            torch.cuda.empty_cache()

        
        lr = optim.param_groups[0]['lr']
        description = f'Step: {step+1}/{len(dataloader)} || Lr: {round(lr, 9)} || Loss: {round(loss.item(), 4)}'
        pbar.set_description(description)
    
    print("Time : ", time.time() - ts)
    print(num_data)
    
    return total_pred, total_label, total_loss / num_data, total_ce_loss / num_data, total_recon_loss / num_recon_data, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer, optim, scheduler

def run_test_epoch(args, data, testdataloader, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer, loss_fn):
    total_pred = []
    total_label = []
    num_data = 0
    num_recon_data = 0
    total_loss = 0
    total_ce_loss = 0
    total_recon_loss = 0
    
    # Batch ==============================================================
    ts = time.time()
    batchcount = 0
    for input_nodes, output_nodes, blocks in testdataloader:      
        # Wrap up loader
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices_in_batch = srcs.to(device)
        nodeindices = blocks[-1].srcdata[dgl.NID]['node'][srcs]
        hedgeindices_in_batch = dsts.to(device)
        hedgeindices = blocks[-1].srcdata[dgl.NID]['edge'][dsts]
        hedgeindices = dsts.to(device)
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)
        
        batchcount += 1
        # Get Embedding
        if args.embedder == "whatsnetLSPE":
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            v_pos = data.v_pos[input_nodes['node']].to(device)
            e_pos = data.e_pos[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat, v_pos, e_pos)
        elif args.embedder == "whatsnet":
            if args.att_type_v in ["ITRE", "ShawRE", "RafRE"]:
                vindex = torch.arange(len(input_nodes['node'])).unsqueeze(1).to(device)
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat, vindex)
            else:
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat)
        else:
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat)
                
        # Predict Class
        if args.scorer == "sm":
          if args.dataset_name == "task1":
            hembedding = e[hedgeindices_in_batch]
            input_embeddings = hembedding
            predictions = scorer(input_embeddings)
          elif args.dataset_name == "task2":
            hembedding = e[hedgeindices_in_batch]
            vembedding = v[nodeindices_in_batch]
            input_embeddings = torch.cat([hembedding,vembedding], dim=1)
            predictions = scorer(input_embeddings)
          else:
            raise ValueError("Not supported dataset type")
          
        elif args.scorer == "im":
            predictions, nodelabels = scorer(blocks[-1], v, e)
        total_pred.append(predictions.detach())
        pred_cls = torch.argmax(predictions, dim=1)
        total_label.append(nodelabels.detach())
        
        num_data += predictions.shape[0]
        num_recon_data += input_nodes['node'].shape[0]
        ce_loss = loss_fn(predictions, nodelabels)
        loss = ce_loss + args.rw * recon_loss

        total_loss += (loss.item() * predictions.shape[0])
        total_ce_loss += (ce_loss.item() * predictions.shape[0])
        total_recon_loss += (recon_loss.item() * input_nodes['node'].shape[0]) # This is fixed as zero
        
    return total_pred, total_label, total_loss / num_data, total_ce_loss / num_data, total_recon_loss / num_recon_data, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer

# Make Output Directory --------------------------------------------------------------------------------------------------------------
initialization = "rw"
args = utils.parse_args()
if args.evaltype == "test":
    assert args.fix_seed
    outputdir = "results_test/" + args.dataset_name + "_" + str(args.k) + "/" + initialization + "/"
    outputParamResFname = outputdir + args.model_name + "/param_result.txt"
    if args.custom:
        outputdir += args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/" + "custom_embedding/"
    else:
        outputdir += args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/"
    if args.recalculate is False and os.path.isfile(outputdir + "log_test_confusion.txt"):
        sys.exit("Already Run")
else:
    outputdir = "results/" + args.dataset_name + "_" + str(args.k) + "/" + initialization + "/"
    outputdir += args.model_name + "/" + args.param_name +"/"
    if args.recalculate is False and os.path.isfile(outputdir + "log_test_confusion.txt"):
        sys.exit("Already Run")
if os.path.isdir(outputdir) is False:
    os.makedirs(outputdir)
print("OutputDir = " + outputdir)

# Initialization --------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset_name = args.dataset_name #'citeseer' 'cora'

if args.fix_seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    dgl.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dgl.seed(args.seed)

exp_num = args.exp_num
test_epoch = args.test_epoch
plot_epoch = args.epochs

if os.path.isfile(outputdir + "checkpoint.pt") and args.recalculate is False:
    print("Start from checkpoint")
elif (args.recalculate is False and args.evaltype == "valid") and os.path.isfile(outputdir + "log_valid_micro.txt"):
    if os.path.isfile(outputdir + "log_valid_micro.txt"):
        max_acc = 0
        cur_patience = 0
        epoch = 0
        with open(outputdir + "log_valid_micro.txt", "r") as f:
            for line in f.readlines():
                ep_str = line.rstrip().split(":")[0].split(" ")[0]
                acc_str = line.rstrip().split(":")[-1]
                epoch = int(ep_str)
                if max_acc < float(acc_str):
                    cur_patience = 0
                    max_acc = float(acc_str)
                else:
                    cur_patience += 1
                if cur_patience > args.patience:
                    break
        if cur_patience > args.patience or epoch == args.epochs:
            sys.exit("Already Run by log valid micro txt")
else:
    if os.path.isfile(outputdir + "log_train.txt"):
        os.remove(outputdir + "log_train.txt")
    if os.path.isfile(outputdir + "log_valid_micro.txt"):
        os.remove(outputdir + "log_valid_micro.txt")
    if os.path.isfile(outputdir + "log_valid_confusion.txt"):
        os.remove(outputdir + "log_valid_confusion.txt")
    if os.path.isfile(outputdir + "log_valid_macro.txt"):
        os.remove(outputdir + "log_valid_macro.txt")
    if os.path.isfile(outputdir + "log_test_micro.txt"):
        os.remove(outputdir + "log_test_micro.txt")
    if os.path.isfile(outputdir + "log_test_confusion.txt"):
        os.remove(outputdir + "log_test_confusion.txt")
    if os.path.isfile(outputdir + "log_test_macro.txt"):
        os.remove(outputdir + "log_test_macro.txt")
        
    if os.path.isfile(outputdir + "initembedder.pt"):
        os.remove(outputdir + "initembedder.pt")
    if os.path.isfile(outputdir + "customerembedder.pt"):
        os.remove(outputdir + "customerembedder.pt")
    if os.path.isfile(outputdir + "colorembedder.pt"):
        os.remove(outputdir + "colorembedder.pt")
    if os.path.isfile(outputdir + "sizeembedder.pt"):
        os.remove(outputdir + "sizeembedder.pt")
    if os.path.isfile(outputdir + "groupembedder.pt"):
        os.remove(outputdir + "groupembedder.pt")
    if os.path.isfile(outputdir + "embedder.pt"):
        os.remove(outputdir + "embedder.pt")
    if os.path.isfile(outputdir + "scorer.pt"):
        os.remove(outputdir + "scorer.pt")
    if os.path.isfile(outputdir + "evaluation.txt"):
        os.remove(outputdir + "evaluation.txt")
            
# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args, dataset_name)
if args.dataset_name == 'task1':
  train_data = data.get_data(0)
  valid_data = data.get_data(1)
elif args.dataset_name == 'task2':
  train_data = data.get_data(0, task2=True)
  valid_data = data.get_data(1, task2=True)
if args.evaltype == "test":
    test_data = data.get_data(2)

ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): args.sampling}] * (args.num_layers * 2 + 1)
full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
if data.weight_flag:
    g = gen_weighted_DGLGraph(args, data.hedge2node, data.hedge2nodePE, data.hedge2nodepos, data.node2hedge, data.node2hedgePE, device)
else:
    g = gen_DGLGraph(args, data.hedge2node, data.hedge2nodepos, data.node2hedge, device)
try:
    sampler = dgl.dataloading.NeighborSampler(ls)
    fullsampler = dgl.dataloading.NeighborSampler(full_ls)
except:
    sampler = dgl.dataloading.MultiLayerNeighborSampler(ls, False)
    fullsampler = dgl.dataloading.MultiLayerNeighborSampler(full_ls)
if args.use_gpu:
    g = g.to(device)
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    if args.evaltype == "test":
        test_data = test_data.to(device)
    data.e_feat = data.e_feat.to(device)
dataloader = dgl.dataloading.DataLoader( g, {"edge": train_data}, sampler, batch_size=args.bs, shuffle=True, drop_last=False) # , num_workers=4
validdataloader = dgl.dataloading.DataLoader( g, {"edge": valid_data}, sampler, batch_size=args.bs, shuffle=True, drop_last=False)
if args.evaltype == "test":
    testdataloader = dgl.dataloading.DataLoader(g, {"edge": test_data}, fullsampler, batch_size=args.bs, shuffle=False, drop_last=False)

args.input_vdim = data.v_feat.size(1)
args.input_edim = data.e_feat.size(1)
args.order_dim = data.order_dim

# init embedder
args.input_vdim = 48
if args.orderflag:
    args.input_vdim = 44
savefname = "../%s_%d_wv_%d_%s.npy" % (args.dataset_name, args.k, args.input_vdim, args.walk)
node_list = np.arange(data.numnodes).astype('int')

# A: (data.numnodes, vector_size)의 embedding 생성, 이는 초기 node embedding으로 사용
if os.path.isfile(savefname) is False:
    walk_path = random_walk_hyper(args, node_list, data.hedge2node)
    walks = np.loadtxt(walk_path, delimiter=" ").astype('int')
    print("Start turning path to strs")
    split_num = 20
    pool = ProcessPoolExecutor(max_workers=split_num)
    process_list = []
    walks = np.array_split(walks, split_num)
    result = []
    for walk in walks:
        process_list.append(pool.submit(utils.walkpath2str, walk))
    for p in as_completed(process_list):
        result += p.result()
    pool.shutdown(wait=True)
    walks = result
    # print(walks)
    print("Start Word2vec")
    print("num cpu cores", multiprocessing.cpu_count())
    w2v = Word2Vec(walks, vector_size=args.input_vdim, window=10, min_count=0, sg=1, epochs=1, workers=multiprocessing.cpu_count())
    print(w2v.wv['0'])
    wv = w2v.wv
    A = [wv[str(i)] for i in range(data.numnodes)]
    np.save(savefname, A)
else:
    print("load exist init walks")
    A = np.load(savefname)
A = StandardScaler().fit_transform(A)
A = A.astype('float32')
A = torch.tensor(A).to(device)
# Make embedding vector with input 
print(data.numnodes, data.numcustomers, data.numcolors, data.numsizes, data.numgroups)
initembedder = Wrap_Embedding(data.numnodes, args.input_vdim, scale_grad_by_freq=False, sparse=False).to(device)
initembedder.weight = nn.Parameter(A)

# Make embedding with feature
# 0-342037
customerembedder = Wrap_Embedding(data.numcustomers, 48, scale_grad_by_freq=False, sparse=False).to(device)
# 0-641
colorembedder = Wrap_Embedding(data.numcolors, 8, scale_grad_by_freq=False, sparse=False).to(device)
# 0-28
sizeembedder = Wrap_Embedding(data.numsizes, 8, scale_grad_by_freq=False, sparse=False).to(device)
# 0-31
groupembedder = Wrap_Embedding(data.numgroups, 8, scale_grad_by_freq=False, sparse=False).to(device)
# Add dimension for embeddings
args.input_vdim = args.input_vdim + 24
# args.input_edim = args.input_edim + 24

print("Model:", args.embedder)
# model init
if args.embedder == "whatsnetLSPE":
    embedder = WhatsnetLSPE(WhatsnetLSPELayer, args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, num_inds=args.num_inds,
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v, att_type_e=args.att_type_e, agg_type_e=args.agg_type_e,
                           num_att_layer=args.num_att_layer, dropout=args.dropout).to(device)
elif args.embedder == "whatsnet":    
    input_vdim = args.input_vdim
    pe_ablation_flag = args.pe_ablation
    embedder = Whatsnet(WhatsnetLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, num_inds=args.num_inds,
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v, att_type_e=args.att_type_e, agg_type_e=args.agg_type_e,
                           num_att_layer=args.num_att_layer, dropout=args.dropout, weight_flag=data.weight_flag, pe_ablation_flag=pe_ablation_flag, vis_flag=args.analyze_att).to(device)
elif args.embedder == "whatsnetHAT":
    input_vdim = args.input_vdim
    embedder = WhatsnetHAT(WhatsnetHATLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, 
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v,
                           num_att_layer=args.num_att_layer, dropout=args.dropout).to(device)
elif args.embedder == "whatsnetHNHN":
    input_vdim = args.input_vdim
    embedder = WhatsnetHNHN(WhatsnetHNHNLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, 
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v,
                           num_att_layer=args.num_att_layer, dropout=args.dropout).to(device)

    
print("Embedder to Device")
print("Scorer = ", args.scorer)
# pick scorer
if args.scorer == "sm":
    if args.dataset_name == 'task1':
      scorer = FC(args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers, args.dropout).to(device)
    elif args.dataset_name == 'task2':
      scorer = FC(args.dim_vertex + args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers, args.dropout).to(device)
    else:
      raise ValueError("Not supported dataset type")
        
elif args.scorer == "im": #whatsnet
    scorer = WhatsnetIM(args.dim_vertex, args.output_dim, dim_hidden=args.dim_hidden, num_layer=args.scorer_num_layers).to(device)

if args.optimizer == "adam":
    optim = torch.optim.Adam(list(initembedder.parameters())+list(customerembedder.parameters())+list(colorembedder.parameters())+list(sizeembedder.parameters())+list(groupembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr) #, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optim = torch.optim.AdamW(list(initembedder.parameters())+list(customerembedder.parameters())+list(colorembedder.parameters())+list(sizeembedder.parameters())+list(groupembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr)
elif args.optimizer == "rms":
    optim = torch.optim.RMSprop(list(initembedder.parameters())+list(customerembedder.parameters())+list(colorembedder.parameters())+list(sizeembedder.parameters())+list(groupembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)

loss_fn = nn.CrossEntropyLoss()

# Train =================================================================================================================================================================================
train_acc=0
patience = 0
best_eval_acc = 0
epoch_start = 1
if os.path.isfile(outputdir + "checkpoint.pt") and args.recalculate is False:
    checkpoint = torch.load(outputdir + "checkpoint.pt") #, map_location=device)
    epoch_start = checkpoint['epoch'] + 1
    initembedder.load_state_dict(checkpoint['initembedder'])
    customerembedder.load_state_dict(checkpoint['customerembedder'])
    colorembedder.load_state_dict(checkpoint['colorembedder'])
    sizeembedder.load_state_dict(checkpoint['sizeembedder'])
    groupembedder.load_state_dict(checkpoint['groupembedder'])
    embedder.load_state_dict(checkpoint['embedder'])
    scorer.load_state_dict(checkpoint['scorer'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_eval_acc = checkpoint['best_eval_acc']
    patience = checkpoint['patience']    
    
    print("Load {} epoch trainer".format(epoch_start))
    print("best_eval_acc = {}\tpatience = {}".format(best_eval_acc, patience))

    if args.save_epochs > 0:
        print("Model Save")
        modelsavename = outputdir + "embedder.pt"
        torch.save(embedder.state_dict(), modelsavename)
        scorersavename = outputdir + "scorer.pt"
        torch.save(scorer.state_dict(), scorersavename)
        initembeddersavename = outputdir + "initembedder.pt"
        torch.save(initembedder.state_dict(),initembeddersavename)
        customerembeddersavename = outputdir + "customerembedder.pt"
        torch.save(customerembedder.state_dict(),customerembeddersavename)
        colorembeddersavename = outputdir + "colorembedder.pt"
        torch.save(colorembedder.state_dict(),colorembeddersavename)
        sizeembeddersavename = outputdir + "sizeembedder.pt"
        torch.save(sizeembedder.state_dict(),sizeembeddersavename)
        groupembeddersavename = outputdir + "groupembedder.pt"
        torch.save(groupembedder.state_dict(),groupembeddersavename)
    
for epoch in tqdm(range(epoch_start, args.epochs + 1), desc='Epoch'): # tqdm
    print("Training")
    
    # Training stage
    initembedder.train()
    customerembedder.train()
    colorembedder.train()
    sizeembedder.train()
    groupembedder.train()
    embedder.train()
    scorer.train()
    # # Calculate Accuracy & Epoch Loss
    total_pred, total_label, train_loss, train_ce_loss, train_recon_loss, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer, optim, scheduler = run_epoch(args, data, dataloader, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer, optim, scheduler, loss_fn, opt="train")
    total_pred = torch.cat(total_pred)
    total_label = torch.cat(total_label, dim=0)
    pred_cls = torch.argmax(total_pred, dim=1)
    train_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
    scheduler.step()
    print("%d epoch: Training loss : %.4f (%.4f, %.4f) / Training acc : %.4f\n" % (epoch, train_loss, train_ce_loss, train_recon_loss, train_acc))
    with open(outputdir + "log_train.txt", "+a") as f:
        f.write("%d epoch: Training loss : %.4f (%.4f, %.4f) / Training acc : %.4f\n" % (epoch, train_loss, train_ce_loss, train_recon_loss, train_acc))
        
    # Test ===========================================================================================================================================================================
    if epoch % test_epoch == 0:
        initembedder.eval()
        customerembedder.eval()
        colorembedder.eval()
        sizeembedder.eval()
        groupembedder.eval()
        embedder.eval()
        scorer.eval()
        
        with torch.no_grad():
            total_pred, total_label, eval_loss, eval_ce_loss, eval_recon_loss, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer, optim, scheduler = run_epoch(args, data, validdataloader, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer, optim, scheduler, loss_fn, opt="valid")
        # Calculate Accuracy & Epoch Loss
        total_label = torch.cat(total_label, dim=0)
        total_pred = torch.cat(total_pred)
        pred_cls = torch.argmax(total_pred, dim=1)
        eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
        y_test = total_label.cpu().numpy()
        pred = pred_cls.cpu().numpy()
        confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='micro', outputdim=args.output_dim)
        with open(outputdir + "log_valid_micro.txt", "+a") as f:
            f.write("{} epoch:Test Loss:{} ({}, {})/Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, eval_loss, eval_ce_loss, eval_recon_loss, accuracy,precision,recall,f1))
        confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='macro', outputdim=args.output_dim)
        with open(outputdir + "log_valid_confusion.txt", "+a") as f:
            for r in range(args.output_dim):
                for c in range(args.output_dim):
                    f.write(str(confusion[r][c]))
                    if c == args.output_dim -1 :
                        f.write("\n")
                    else:
                        f.write("\t")
        with open(outputdir + "log_valid_macro.txt", "+a") as f:               
            f.write("{} epoch:Test Loss:{} ({}, {})/Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, eval_loss, eval_ce_loss, eval_recon_loss, accuracy,precision,recall,f1))

        if best_eval_acc < eval_acc:
            print(best_eval_acc)
            best_eval_acc = eval_acc
            patience = 0
            if args.evaltype == "test" or args.save_epochs > 0:
                print("Model Save")
                modelsavename = outputdir + "embedder.pt"
                torch.save(embedder.state_dict(), modelsavename)
                scorersavename = outputdir + "scorer.pt"
                torch.save(scorer.state_dict(), scorersavename)
                initembeddersavename = outputdir + "initembedder.pt"
                torch.save(initembedder.state_dict(),initembeddersavename)
                customerembeddersavename = outputdir + "customerembedder.pt"
                torch.save(customerembedder.state_dict(),customerembeddersavename)
                colorembeddersavename = outputdir + "colorembedder.pt"
                torch.save(colorembedder.state_dict(),colorembeddersavename)
                sizeembeddersavename = outputdir + "sizeembedder.pt"
                torch.save(sizeembedder.state_dict(),sizeembeddersavename)
                groupembeddersavename = outputdir + "groupembedder.pt"
                torch.save(groupembedder.state_dict(),groupembeddersavename)
        else:
            patience += 1

        if patience > args.patience:
            break
        
        torch.save({
            'epoch': epoch,
            'embedder': embedder.state_dict(),
            'scorer' : scorer.state_dict(),
            'initembedder' : initembedder.state_dict(),
            'customerembedder': customerembedder.state_dict(),
            'colorembedder': colorembedder.state_dict(),
            'sizeembedder': sizeembedder.state_dict(),
            'groupembedder': groupembedder.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'optimizer': optim.state_dict(),
            'best_eval_acc' : best_eval_acc,
            'patience' : patience
            }, outputdir + "checkpoint.pt")

# if args.evaltype == "test":
#     print("Test")
    
#     initembedder.load_state_dict(torch.load(outputdir + "initembedder.pt")) # , map_location=device
#     embedder.load_state_dict(torch.load(outputdir + "embedder.pt")) # , map_location=device
#     scorer.load_state_dict(torch.load(outputdir + "scorer.pt")) # , map_location=device
    
#     initembedder.eval()
#     customerembedder.eval()
#     colorembedder.eval()
#     sizeembedder.eval()
#     groupembedder.eval()
#     embedder.eval()
#     scorer.eval()

#     with torch.no_grad():
#         total_pred, total_label, test_loss, test_ce_loss, test_recon_loss, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer = run_test_epoch(args, data, testdataloader, initembedder, customerembedder, colorembedder, sizeembedder, groupembedder, embedder, scorer, loss_fn)
#     # Calculate Accuracy & Epoch Loss
#     total_label = torch.cat(total_label, dim=0)
#     total_pred = torch.cat(total_pred)
#     pred_cls = torch.argmax(total_pred, dim=1)
#     eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
#     y_test = total_label.cpu().numpy()
#     pred = pred_cls.cpu().numpy()
#     confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='micro', outputdim=args.output_dim)
#     with open(outputdir + "log_test_micro.txt", "+a") as f:
#         f.write("{} epoch:Test Loss:{} ({}, {})/Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, test_loss, test_ce_loss, test_recon_loss, accuracy,precision,recall,f1))
#     confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='macro', outputdim=args.output_dim)
#     with open(outputdir + "log_test_confusion.txt", "+a") as f:
#         for r in range(args.output_dim):
#             for c in range(args.output_dim):
#                 f.write(str(confusion[r][c]))
#                 if c == args.output_dim -1 :
#                     f.write("\n")
#                 else:
#                     f.write("\t")
#     with open(outputdir + "log_test_macro.txt", "+a") as f:               
#         f.write("{} epoch:Test Loss:{} ({}, {})/Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, test_loss, test_ce_loss, test_recon_loss, accuracy,precision,recall,f1))

# if os.path.isfile(outputdir + "checkpoint.pt"):
#     os.remove(outputdir + "checkpoint.pt")

