# AI506 Term Project: E-commerce Product Return Prediction

> <b>KAIST 2024 Spring AI506: Data Mining and Search</b>

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

