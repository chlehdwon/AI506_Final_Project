# AI506 Term Project: E-commerce Product Return Prediction

> <b>KAIST 2024 Spring AI506: Data Mining and Search</b>

> ABCDE-NET: **A**ttention **B**lock with **C**ustomer **D**ependency for **E**-commerce **NET**work

<img width="1264" alt="image" src="https://github.com/chlehdwon/AI506_Final_Project/assets/68576681/953f001d-01ed-4622-b8e0-544cfa0c5bdb">

## Project Summary
The ultimate goal of this project is to practice data mining research by predicting product returns in e-commerce settings.

Our method is designed to solve 2 E-commerce Product Return Prediction tasks: (1) Order Return Prediction and (2) Product Return Prediction.

It models complex relations between orders and products as **hypergraphs** based on the **WHATsNET** paper: [Classification of Edge-Dependent Labels of Nodes in Hypergraphs](https://arxiv.org/abs/2306.03032)

The contributions of this project are the following:

- Our proposed method is a effective method for prediction of product returns based on hypergraphs.
- It is effective and comparable to naive methods such as random guessing, SVM, or random forests.
- Initialization of node embeddings from product features (color, size, product group) which are concatenated to random walk initialization.
- Implementation of Customer-dependent MAB for consideration of customer preference on product return.

## Team Information

- [Dongwon Choi](https://github.com/chlehdwon) (School of Computing, KAIST)
- [Sangwoo Kim](https://github.com/lodikim) (Graduate School of AI, KAIST)
- [Dohun Lee](https://github.com/DoHunLee1) (Graduate School of AI, KAIST)

## Environment

- Python >= 3.8
- Pytorch: 2.2.1
- Cuda: 11.8
- GPU: RTX 4090

## How to run

### Train

You can train our model for each task by following commands:

```bash:train.sh
// Task 1
python train.py --dataset_name task1 --output_dim 3 --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 1024 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 1 --evaltype test --save_epochs 1 --seed 42 --fix_seed --recalculate --use_wandb=1 --custom

// Task 2
python train.py --dataset_name task2 --output_dim 2 --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 1024 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 1 --evaltype test --save_epochs 1 --seed 42 --fix_seed --recalculate --use_wandb=1 --custom
```

### Predict

You can produce predictions from our model for each task by following commands:

```bash:predict.sh
// Task 1
python predict.py --dataset_name task1 --output_dim 3 --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 1024 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 1 --evaltype test --save_epochs 1 --seed 42 --fix_seed --use_wandb=0 --custom

// Task 2
python predict.py --dataset_name task2 --output_dim 2 --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 1024 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 1 --evaltype test --save_epochs 1 --seed 42 --fix_seed --use_wandb=0 --custom
```
