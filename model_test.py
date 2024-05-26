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
import time
import argparse
import dgl
import pickle
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

from model.HNHN import HNHN
from model.HGNN import HGNN
from model.HAT import HyperAttn
from model.UniGCN import UniGCNII
from model.Whatsnet import Whatsnet, WhatsnetLayer
from model.WhatsnetClassifier import WhatsnetClassifier
from model.layer import FC, Wrap_Embedding


args = utils.argparse()
device = torch.device("cuda" if torch.cuda.is_availabel() else "cpu")
input_vdim = args.input_vdim
pe_ablation_flag = args.pe_ablation
embedder = Whatsnet(WhatsnetLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                        weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, num_inds=args.num_inds,
                        att_type_v=args.att_type_v, agg_type_v=args.agg_type_v, att_type_e=args.att_type_e, agg_type_e=args.agg_type_e,
                        num_att_layer=args.num_att_layer, dropout=args.dropout, weight_flag=data.weight_flag, pe_ablation_flag=pe_ablation_flag, vis_flag=args.analyze_att).to(device)