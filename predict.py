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
from model.WhatsnetHAT import WhatsnetHAT, WhatsnetHATLayer
from model.WhatsnetHNHN import WhatsnetHNHN, WhatsnetHNHNLayer
from model.layer import FC, Wrap_Embedding

# Make Output Directory --------------------------------------------------------------------------------------------------------------
initialization = "rw"
args = utils.parse_args()
outputdir = "results_test/" + args.dataset_name + "_" + str(args.k) + "/" + initialization + "/"
outputdir += args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/" + "custom_embedding/"
if os.path.isdir(outputdir) is False:
    os.makedirs(outputdir)
print("OutputDir = " + outputdir)

# Initialization --------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset_name #'citeseer' 'cora'
opt = 'valid'
print(f'Device: {device}, Dataset name: {dataset_name}, Option: {opt}')

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


# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args, dataset_name)
if args.dataset_name == 'task1':
  valid_data = data.get_data(1)
  test_data = data.get_data(2)
elif args.dataset_name == 'task2':
  valid_data = data.get_data(1, task2=True)
  test_data = data.get_data(2, task2=True)

#allhedges = torch.LongTensor(np.arange(data.numhedges))
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
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    data.e_feat = data.e_feat.to(device)
    #hedge_data = allhedges.to(device)
#else:
    #hedge_data = allhedges
#dataloader = dgl.dataloading.NodeDataLoader( g, {"edge": hedge_data}, fullsampler, batch_size=args.bs, shuffle=False, drop_last=False) # , num_workers=4
if opt == 'valid':
    dataloader = dgl.dataloading.DataLoader(g, {"edge": valid_data}, sampler, batch_size=args.bs, shuffle=True, drop_last=False)
else:
    dataloader = dgl.dataloading.DataLoader(g, {"edge": test_data}, fullsampler, batch_size=args.bs, shuffle=False, drop_last=False)

args.input_vdim = data.v_feat.size(1)
args.input_edim = data.e_feat.size(1)
args.order_dim = data.order_dim

# init embedder
args.input_vdim = 48
if args.orderflag:
    args.input_vdim = 44
#args.input_edim = data.e_feat.size(1)
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
    w2v = Word2Vec( walks, vector_size=args.input_vdim, window=10, min_count=0, sg=1, epochs=1, workers=multiprocessing.cpu_count())
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
initembedder = Wrap_Embedding(data.numnodes, args.input_vdim, scale_grad_by_freq=False, padding_idx=0, sparse=False)
initembedder.weight = nn.Parameter(A)

# Make embedding with feature
# 0-342037
customerembedder = Wrap_Embedding(data.numcustomers, 24, scale_grad_by_freq=False, sparse=False).to(device)
# 0-641
colorembedder = Wrap_Embedding(data.numcolors, 8, scale_grad_by_freq=False, sparse=False).to(device)
# 0-28
sizeembedder = Wrap_Embedding(data.numsizes, 8, scale_grad_by_freq=False, sparse=False).to(device)
# 0-31
groupembedder = Wrap_Embedding(data.numgroups, 8, scale_grad_by_freq=False, sparse=False).to(device)
# Add dimension for embeddings
args.input_vdim = args.input_vdim + 24
args.input_edim = args.input_edim + 24

print("Model:", args.embedder)
# model init
if args.embedder == "whatsnet":    
    input_vdim = args.input_vdim
    pe_ablation_flag = args.pe_ablation
    embedder = Whatsnet(WhatsnetLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, num_inds=args.num_inds,
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v, att_type_e=args.att_type_e, agg_type_e=args.agg_type_e,
                           num_att_layer=args.num_att_layer, dropout=args.dropout, weight_flag=data.weight_flag, pe_ablation_flag=pe_ablation_flag).to(device)
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
#if args.scorer == "sm":
#    scorer = FC(args.dim_vertex + args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers, args.dropout).to(device)

if args.scorer == "sm":
    if args.dataset_name == 'task1':
      scorer = FC(args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers, args.dropout).to(device)
    elif args.dataset_name == 'task2':
      scorer = FC(args.dim_vertex + args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers, args.dropout).to(device)
    else:
      raise ValueError("Not supported dataset type")


if args.optimizer == "adam":
    optim = torch.optim.Adam(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr) #, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optim = torch.optim.AdamW(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr)
elif args.optimizer == "rms":
    optime = torch.optim.RMSprop(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
loss_fn = nn.CrossEntropyLoss()

# Load Checkpoints ===========================================================================================================================================================================
initembedder.load_state_dict(torch.load(outputdir + "initembedder.pt")) # , map_location=device
embedder.load_state_dict(torch.load(outputdir + "embedder.pt")) # , map_location=device
scorer.load_state_dict(torch.load(outputdir + "scorer.pt")) # , map_location=device

customerembedder.load_state_dict(torch.load(outputdir + "customerembedder.pt")) # , map_location=device
colorembedder.load_state_dict(torch.load(outputdir + "colorembedder.pt")) # , map_location=device
sizeembedder.load_state_dict(torch.load(outputdir + "sizeembedder.pt")) # , map_location=device
groupembedder.load_state_dict(torch.load(outputdir + "groupembedder.pt")) # , map_location=device

# Test ===========================================================================================================================================================================
initembedder.eval()
embedder.eval()
scorer.eval()

customerembedder.eval()
colorembedder.eval()
sizeembedder.eval()
groupembedder.eval()

with torch.no_grad():
    allpredictions = defaultdict(dict)
    
    total_pred = []
    total_label = []
    #num_data = 0
    
    # Batch ==============================================================
    for input_nodes, output_nodes, blocks in tqdm(dataloader): #, desc="batch"):      
        # Wrap up loader
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices = srcs.to(device)
        hedgeindices = dsts.to(device)
        #nodeindices = blocks[-1].srcdata[dgl.NID]['node'][srcs]
        #hedgeindices = blocks[-1].srcdata[dgl.NID]['edge'][dsts]
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)
        
        # Get Embedding
        if args.embedder == "whatsnet":
            if args.att_type_v in ["ITRE", "ShawRE", "RafRE"]:
                vindex = torch.arange(len(input_nodes['node'])).unsqueeze(1).to(device)
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat, vindex)
            else:
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                # concat feature embedding
                customer_feat, _ = customerembedder(data.hedge2customer[input_nodes['edge']].long().to(device))
                color_feat, _ = colorembedder(data.node2color[input_nodes['node']].long().to(device))
                size_feat, _ = sizeembedder(data.node2size[input_nodes['node']].long().to(device))
                group_feat, _ = groupembedder(data.node2group[input_nodes['node']].long().to(device))
                v_feat = torch.concat((v_feat, color_feat, size_feat, group_feat), dim=1)
                e_feat = torch.concat((e_feat, customer_feat), dim=1)
                v, e = embedder(blocks, v_feat, e_feat)
        else:
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat)
                
        # Predict Class
        '''
        hembedding = e[hedgeindices_in_batch]
        vembedding = v[nodeindices_in_batch]
        input_embeddings = torch.cat([hembedding,vembedding], dim=1)
        predictions = scorer(input_embeddings)
        
        total_pred.append(predictions.detach())
        pred_cls = torch.argmax(predictions, dim=1)
        total_label.append(nodelabels.detach())
        '''
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
                if opt == 'valid':
                    valid_indices = torch.nonzero((nodelabels == 2) | (nodelabels == 3)).squeeze()
                    hedgeindices = hedgeindices[valid_indices]
                    nodeindices = nodeindices[valid_indices]
                    nodelabels = nodelabels[valid_indices]
                    nodelabels = nodelabels - 2

                    hembedding = e[hedgeindices]
                    vembedding = v[nodeindices]
                    input_embeddings = torch.cat([hembedding,vembedding], dim=1)
                    predictions = scorer(input_embeddings)
                elif opt == 'test':
                    test_indices = torch.nonzero((nodelabels == -1)).squeeze()
                    hedgeindices = hedgeindices[test_indices]
                    nodeindices = nodeindices[test_indices]
                    nodelabels = nodelabels[test_indices]
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
        
        '''
        for v, h, vpred, vlab in zip(nodeindices.tolist(), hedgeindices.tolist(), pred_cls.detach().cpu().tolist(), nodelabels.detach().cpu().tolist()):
            assert v in data.hedge2node[h]
            for vorder in range(len(data.hedge2node[h])):
                if data.hedge2node[h][vorder] == v:
                    assert vlab == data.hedge2nodepos[h][vorder]
            if args.binning > 0:
                allpredictions[h][v] = data.binindex[int(vpred)]
            elif args.dataset_name == "Etail":
                allpredictions[h][v] = int(vpred) + 1.0
            else:
                allpredictions[h][v] = int(vpred)
        num_data += predictions.shape[0]
        '''

    # Calculate Accuracy (For Validation)    
    total_label = torch.cat(total_label, dim=0)
    total_pred = torch.cat(total_pred)
    pred_cls = torch.argmax(total_pred, dim=1)
    eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
    y_test = total_label.cpu().numpy()
    pred = pred_cls.cpu().numpy()
    confusion, accuracy, precision, recall, f1_micro = utils.get_clf_eval(y_test, pred, avg='micro', outputdim=args.output_dim)
    with open(outputdir + "all_micro.txt", "w") as f:
        f.write("Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(accuracy,precision,recall,f1_micro))
    confusion, accuracy, precision, recall, f1_macro = utils.get_clf_eval(y_test, pred, avg='macro', outputdim=args.output_dim)
    with open(outputdir + "all_confusion.txt", "w") as f:
        for r in range(args.output_dim):
            for c in range(args.output_dim):
                f.write(str(confusion[r][c]))
                if c == args.output_dim -1 :
                    f.write("\n")
                else:
                    f.write("\t")
    with open(outputdir + "all_macro.txt", "w") as f:               
        f.write("Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(accuracy,precision,recall,f1_macro))
        
    with open(outputdir + "prediction.txt", "w") as f:
        for h in range(data.numhedges):
            line = []
            for v in data.hedge2node[h]:
                line.append(str(allpredictions[h][v]))
            f.write("\t".join(line) + "\n")
            
    # output
    if os.path.isdir("train_results/{}/".format(args.dataset_name)) is False:
        os.makedirs("train_results/{}/".format(args.dataset_name))
    with open("train_results/{}/prediction_{}_{}.txt".format(args.dataset_name, args.outputname, args.seed), "w") as f:
        for h in range(data.numhedges):
            line = []
            for v in data.hedge2node[h]:
                line.append(str(allpredictions[h][v]))
            f.write("\t".join(line) + "\n")
            
    if os.path.isdir("train_results/{}/f1/".format(args.dataset_name)) is False:
        os.makedirs("train_results/{}/f1/".format(args.dataset_name))
    if os.path.isfile("train_results/{}/f1/{}.txt".format(args.dataset_name, args.outputname)) is False:
        with open("train_results/{}/f1/{}.txt".format(args.dataset_name, args.outputname), "w") as f:
            f.write("seed,f1micro,f1macro\n")
    with open("train_results/{}/f1/{}.txt".format(args.dataset_name, args.outputname), "+a") as f:
        f.write(",".join([str(args.seed), str(f1_micro), str(f1_macro)]) + "\n")
 