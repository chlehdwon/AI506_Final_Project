import torch
import numpy as np
import math
from collections import defaultdict
import dgl
import utils
import random
import pickle
import os
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import train_test_split
from dgl import dataloading
from dgl import sampling, subgraph, distributed
from tqdm import tqdm, trange
import scipy.sparse as sp
import hashlib
from scipy.sparse.linalg import expm
from scipy.sparse import csr_matrix, diags

def make_order(ls):
    a = np.array(ls)
    argsorted = np.argsort(a)
    orders = np.zeros(a.shape)
    a = sorted(a)
    
    # adjust
    previous = None
    order = 1
    for i, _a  in enumerate(a):
        if previous is None:
            previous = _a
            orders[argsorted[i]] = order
        elif previous != _a:
            order += 1
            orders[argsorted[i]] = order
            previous = _a
        else:
            orders[argsorted[i]] = order
    return orders.tolist()

class Hypergraph:
    def __init__(self, args, dataset_name):
        self.inputdir = args.inputdir
        self.dataname = dataset_name
        self.exist_hedgename = args.exist_hedgename
        self.valid_inputname = args.valid_inputname
        self.test_inputname = args.test_inputname
        self.use_gpu = args.use_gpu
        self.k = args.k
        
        self.hedge2node = []
        self.node2hedge = [] 
        self.hedge2nodepos = [] # hyperedge index -> node positions (after binning)
        self._hedge2nodepos = [] # hyperedge index -> node positions (before binning)
        self.node2hedgePE = []
        self.hedge2nodePE = []
        self.weight_flag = False
        self.hedge2nodeweight = []
        self.node2hedgeweight = []
        self.numhedges = 0
        self.numnodes = 0
        
        self.hedge2customer = []
        self.node2color = []
        self.node2size = []
        self.node2group = []
        self.numcustomers = 0
        self.numcolors = 0
        self.numsizes = 0
        self.numgroups = 0
        
        self.hedgeindex = {} # papaercode -> index
        self.hedgename = {} # index -> papercode
        self.e_feat = []

        self.node_reindexing = {} # nodeindex -> reindex
        self.node_orgindex = {} # reindex -> nodeindex
        self.v_feat = [] # (V, 1)
        
        self.load_graph(args)        
        print("Data is prepared")
        
    def load_graph(self, args):
        # construct connection  -------------------------------------------------------
        hset = []
        if args.k > 0:
            with open(self.inputdir + self.dataname + "/sampled_hset_" + str(args.k) + ".txt", "r") as f:
                for line in f.readlines():
                    line = line.rstrip()
                    hset.append(int(line))
        self.max_len = 0
        with open(self.inputdir + self.dataname + "/hypergraph.txt", "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                if (args.k == 0) or ((args.k > 0) and (_hidx in hset)):
                    tmp = line.split("\t")
                    hidx = self.numhedges
                    self.numhedges += 1
                    self.hedgeindex[_hidx] = hidx
                    self.hedgename[hidx] = _hidx
                    self.hedge2node.append([])
                    self.hedge2nodepos.append([])
                    self._hedge2nodepos.append([])
                    self.hedge2nodePE.append([])
                    self.hedge2nodeweight.append([])
                    self.e_feat.append([])
                    if (self.max_len < len(tmp)):
                        self.max_len = len(tmp)
                    for node in tmp:
                        node = int(node.rstrip())
                        if node not in self.node_reindexing:
                            node_reindex = self.numnodes
                            self.numnodes += 1 
                            self.node_reindexing[node] = node_reindex
                            self.node_orgindex[node_reindex] = node 
                            self.node2hedge.append([])
                            self.node2hedgePE.append([])
                            self.node2hedgeweight.append([])
                            self.v_feat.append([])
                        nodeindex = self.node_reindexing[node]
                        self.hedge2node[hidx].append(nodeindex)
                        self.node2hedge[nodeindex].append(hidx)
                        self.hedge2nodePE[hidx].append([])
                        self.node2hedgePE[nodeindex].append([])
                    
        print("Max Size = ", self.max_len)
        print("Number of Hyperedges : " + str(self.numhedges))
        print("Number of Nodes : " + str(self.numnodes))
        # update by max degree
        for vhedges in self.node2hedge:
            if self.max_len < len(vhedges):
                self.max_len = len(vhedges)
        self.v_feat = torch.tensor(self.v_feat).type('torch.FloatTensor')
        for h in range(len(self.e_feat)):
            self.e_feat[h] = [0 for _ in range(args.dim_edge)]
        self.e_feat = torch.tensor(self.e_feat).type('torch.FloatTensor')
        
        # Split Data ------------------------------------------------------------------------
        self.test_index = []
        self.valid_index = []
        self.validsize = 0
        self.testsize = 0
        self.trainsize = 0
        self.hedge2type = torch.zeros(self.numhedges)
        
        assert os.path.isfile(self.inputdir + self.dataname + "/" + self.valid_inputname + "_" + str(self.k) + ".txt")
        with open(self.inputdir + self.dataname + "/" + self.valid_inputname + "_" + str(self.k) + ".txt", "r") as f:
            for line in f.readlines():
                name = line.rstrip()
                if self.exist_hedgename is False:
                    name = int(name)
                index = self.hedgeindex[name]
                self.valid_index.append(index)
            self.hedge2type[self.valid_index] = 1
            self.validsize = len(self.valid_index)
        if os.path.isfile(self.inputdir + self.dataname + "/" + self.test_inputname + "_" + str(self.k) + ".txt"):
            with open(self.inputdir + self.dataname + "/" + self.test_inputname + "_" + str(self.k) + ".txt", "r") as f:
                for line in f.readlines():
                    name = line.rstrip()
                    if self.exist_hedgename is False:
                        name = int(name)
                    index = self.hedgeindex[name]
                    self.test_index.append(index)
                assert len(self.test_index) > 0
                self.hedge2type[self.test_index] = 2
                self.testsize = len(self.test_index)
        self.trainsize = self.numhedges - (self.validsize + self.testsize)
        
        # extract target ---------------------------------------------------------
        print("Extract labels")
        with open(self.inputdir + self.dataname + "/hypergraph_pos.txt", "r") as f:
            for _hidx, line in enumerate(f.readlines()):
                tmp = line.split("\t")
                if self.exist_hedgename:
                    papercode = tmp[0][1:-1] # without ''
                    if (papercode not in self.hedgeindex):
                        continue
                    hidx = self.hedgeindex[papercode]
                    tmp = tmp[1:]
                else:
                    if (_hidx not in self.hedgeindex):
                        continue
                    hidx = self.hedgeindex[_hidx]
                if args.binning > 0:
                    positions = [float(i) for i in tmp]
                    for nodepos in positions:
                        self._hedge2nodepos[hidx].append(nodepos)
                else:
                    positions = [int(i) for i in tmp]
                for nodepos in positions:
                    self.hedge2nodepos[hidx].append(nodepos)
        # extract PE ----------------------------------------------------------------------------------------------------
        # hedge2nodePE
        if len(args.vorder_input) > 0: # centrality -> PE ------------------------------------------------------------------
            self.order_dim = len(args.vorder_input)
            for inputpath in args.vorder_input:
                vfeat = {} # node -> vfeat
                with open(self.inputdir + self.dataname + "/" + inputpath + "_" + str(args.k) + ".txt", "r") as f:
                    columns = f.readline()
                    columns = columns[:-1].split("\t")
                    for line in f.readlines():
                        line = line.rstrip()
                        tmp = line.split("\t")
                        nodeindex = int(tmp[0])
                        if nodeindex not in self.node_reindexing:
                            # not include in incidence matrix
                            continue
                        node_reindex = self.node_reindexing[nodeindex]
                        for i, col in enumerate(columns):
                            vfeat[node_reindex] = float(tmp[i])
                if args.whole_order: # in entire nodeset
                    feats = []
                    for vidx in range(self.numnodes):
                        feats.append(vfeat[vidx])
                    orders = make_order(feats)
                    for hidx, hedge in enumerate(self.hedge2node):
                        for vorder, v in enumerate(hedge):
                            self.hedge2nodePE[hidx][vorder].append((orders[v]) / self.numnodes)
                else: # in each hyperedge
                    for hidx, hedge in enumerate(self.hedge2node):
                        feats = []
                        for v in hedge:
                            feats.append(vfeat[v])
                        orders = make_order(feats)
                        for vorder, v in enumerate(hedge):
                            self.hedge2nodePE[hidx][vorder].append((orders[vorder]) / len(feats))
            # check            
            assert len(self.hedge2nodePE) == self.numhedges
            for hidx in range(self.numhedges):
                assert len(self.hedge2nodePE[hidx]) == len(self.hedge2node[hidx])
                for vorder in self.hedge2nodePE[hidx]:
                    assert len(vorder) == len(args.vorder_input)
            # node2hedgePE
            for vidx, node in enumerate(self.node2hedge):
                orders = []
                for hidx in node:
                    for vorder,_v in enumerate(self.hedge2node[hidx]):
                        if _v == vidx:
                            orders.append(self.hedge2nodePE[hidx][vorder])
                            break
                self.node2hedgePE[vidx] = orders
            # check
            assert len(self.node2hedgePE) == self.numnodes
            for vidx in range(self.numnodes):
                assert len(self.node2hedgePE[vidx]) == len(self.node2hedge[vidx])
                for horder in self.node2hedgePE[vidx]:
                    assert len(horder) == len(args.vorder_input)
            self.weight_flag = True
        
        # extract feature ----------------------------------------------------------------------------------------------------
        print("Extract features")
        with open(self.inputdir + self.dataname + "/" + self.dataname + "_data.txt", "r") as f:
            f.readline()
            hedge2customer, node2color, node2size, node2group,  = {}, {}, {}, {}
            
            for line in f.readlines():
                order, product, customer, color, size, group = map(int, line.strip().split(','))
                hidx, vidx = order, self.node_reindexing[product]
                if hidx not in hedge2customer.keys():
                    hedge2customer[hidx] = customer
                if vidx not in node2color.keys():
                    node2color[vidx] = color
                if vidx not in node2size.keys():
                    node2size[vidx] = size
                if vidx not in node2group.keys():
                    node2group[vidx] = group

            self.hedge2customer, self.numcustomers = [hedge2customer[idx] for idx in range(len(hedge2customer.keys()))], max(hedge2customer.values()) + 1
            self.node2color, self.numcolors = [node2color[idx] for idx in range(len(node2color.keys()))], max(node2color.values()) + 1
            self.node2size, self.numsizes = [node2size[idx] for idx in range(len(node2size.keys()))], max(node2size.values()) + 1
            self.node2group, self.numgroups = [node2group[idx] for idx in range(len(node2group.keys()))], max(node2group.values()) + 1
            
            self.hedge2customer = torch.Tensor(self.hedge2customer)
            self.node2color = torch.Tensor(self.node2color)
            self.node2size = torch.Tensor(self.node2size)
            self.node2group = torch.Tensor(self.node2group)

    def get_data(self, type=0, task2=False):
        if task2:
          hedgelist = ((self.hedge2type >= type).nonzero(as_tuple=True)[0])
        else:  
          hedgelist = ((self.hedge2type == type).nonzero(as_tuple=True)[0])
        if self.use_gpu is False:
            hedgelist = hedgelist.tolist()
        return hedgelist
    
# Generate DGL Graph ==============================================================================================
def gen_DGLGraph(args, hedge2node, hedge2nodepos, node2hedge, device):
    data_dict = defaultdict(list)
    in_edge_label = []
    con_edge_label = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            in_edge_label.append(hedge2nodepos[hidx][vorder]) 
            con_edge_label.append(hedge2nodepos[hidx][vorder]) 

    in_edge_label = torch.Tensor(in_edge_label)
    con_edge_label = torch.Tensor(con_edge_label)

    g = dgl.heterograph(data_dict)
    g['in'].edata['label'] = in_edge_label
    g['con'].edata['label'] = con_edge_label
    return g

def gen_weighted_DGLGraph(args, hedge2node, hedge2nodePE, hedge2nodepos, node2hedge, node2hedgeorder, device):
    edgefeat_dim = 0
    for efeat_list in hedge2nodePE:
        efeat_dim = len(efeat_list[0])
        edgefeat_dim = max(edgefeat_dim, efeat_dim)
    print("Edge Feat Dim ", edgefeat_dim)
    
    data_dict = defaultdict(list)
    in_edge_weights = []
    in_edge_label = []
    con_edge_weights = []
    con_edge_label = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):

            # connection
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            # edge feat
            efeat = hedge2nodePE[hidx][vorder]
            efeat += np.zeros(edgefeat_dim - len(efeat)).tolist()
            in_edge_weights.append(efeat)
            con_edge_weights.append(efeat)
            # label
            in_edge_label.append(hedge2nodepos[hidx][vorder])
            con_edge_label.append(hedge2nodepos[hidx][vorder])

    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)
    in_edge_label = torch.Tensor(in_edge_label)
    con_edge_label = torch.Tensor(con_edge_label)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
    g['in'].edata['label'] = in_edge_label
    g['con'].edata['label'] = con_edge_label
    
    return g
