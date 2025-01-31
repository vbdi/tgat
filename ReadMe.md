

# Exact Combinatorial Optimization with Temporo-Attentional Graph Neural Networks

This code provides a PyTorch implementation for **TGAT**, as described in the paper ''Exact Combinatorial Optimization with Temporo-Attentional Graph Neural Networks'', published at ECML2023.

Link to paper: https://arxiv.org/abs/2311.13843

Link to Huawei's AI Gallery Notebook: [https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=58b799a0-5cfc-4c2e-8b9b-440bb2315264](https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=047c6cf2-8463-40d7-b92f-7b2ca998e935)

Combinatorial optimization finds an optimal solution within a discrete set of variables and constraints. The field has seen a tremendous progress both in research and industry. With the success of deep learning in the past decade, a recent trend in combinatorial optimization has been to improve state-of-the-art combinatorial optimization solvers by replacing key heuristic components with machine learning (ML) models. In this paper, we investigate two essential aspects of machine learning algorithms for combinatorial optimization: temporal characteristics and attention. We argue that for the task of variable selection in the branch-and-bound (B\&B) algorithm, incorporating the temporal information as well as the bipartite graph attention improves the solver's performance.

<img src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/tgat/system_module.png" style="zoom: 67%;" />

## Running  training

### Requirements

- python 3.6
- pytorch 1.10
- torchvision
- CUDA 10.2
- Other requirements can be found in [ml4co competition](https://github.com/ds4dm/learn2branch-ecole) 

### Training

multi GPU TGAT training:

```bash
python -m torch.distributed.launch --nproc_per_node=[num_gpu_nodes] 03_train_tgat.py --distributed True --samples_path [data_path] --version [arbitray folder to save ckp] --problem [problem] --pytorch_gat_II --num_heads_per_layer [num_attn_heads] --GRU_temp_dim [gru_sequence_length] --emb_size [embedded_size]

```

multi GPU GAT training:

```bash
python -m torch.distributed.launch --nproc_per_node=[num_gpu_nodes] 03_train_gnn.py --distributed True --samples_path [data_path] --version [arbitray folder to save ckp] --problem [problem] --pytorch_gat_II --num_heads_per_layer [num_attn_heads] --emb_size [embedded_size]

```

multi GPU GCNN training:

```bash
python -m torch.distributed.launch --nproc_per_node=[num_gpu_nodes] 03_train_gnn.py --distributed True --samples_path [data_path] --version [arbitray folder to save ckp] --problem [problem] --emb_size [embedded_size]

```

### Dual Integral Reward Evaluation 

To evaluate dual integral reward for TGAT run:

```bash
python 05_evaluate_reward_tgat.py --samples_path [data_path] --version [arbitray folder to save ckp] --problem [problem] --pytorch_gat_II --num_heads_per_layer [num_attn_heads] --emb_size [embedded_size] --seed[seed]

```

To evaluate dual integral reward for GAT run:

```bash
python 05_evaluate_reward.py --samples_path [data_path] --version [arbitray folder to save ckp] --problem [problem] --pytorch_gat_II --num_heads_per_layer [num_attn_heads] --emb_size [embedded_size] --seed[seed]

```
