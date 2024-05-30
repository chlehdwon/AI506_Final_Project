# DefaultArgs, Task 1
CUDA_VISIBLE_DEVICES=3 python predict.py --dataset_name task1 --output_dim 3 --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 1024 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 1 --evaltype test --save_epochs 1 --seed 42 --fix_seed --use_wandb=0 --run_name='DefaultArgs'

# DefaultArgs, Task 2
#CUDA_VISIBLE_DEVICES=3 python predict.py --dataset_name task2 --output_dim 2 --vorder_input "degree_nodecentrality,eigenvec_nodecentrality,pagerank_nodecentrality,kcore_nodecentrality" --embedder whatsnet --att_type_v OrderPE --agg_type_v PrevQ --att_type_e OrderPE --agg_type_e PrevQ --num_att_layer 2 --num_layers 1 --scorer sm --scorer_num_layers 1 --bs 1024 --lr 0.001 --sampling 40 --dropout 0.7 --optimizer "adam" --k 0 --gamma 0.99 --dim_hidden 64 --dim_edge 128 --dim_vertex 128 --epochs 100 --test_epoch 1 --evaltype test --save_epochs 1 --seed 42 --fix_seed --use_wandb=0 --run_name='DefaultArgs'