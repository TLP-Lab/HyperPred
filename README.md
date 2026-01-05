## 1. Overview

PyTorch Implementation for "HyperPred: Multi-Scale Hyperbolic Framework for Scalable Temporal Link Prediction on Dynamic Graphs"

Authors: Fan Wu, QiYuan Wu, Hao Wu, Zhaoyun Ding, Chao Chen,Shuang Gu,Feng Lyu


## 2. Examples

Run `python main.py --dataset=dblp` for example.

**Some critical config parameters:** 

- `--dataset`, `--data_pt_path`: name and parent path of dataset
- `--test_length`: number of snapshots for test set
- `--diffusion_steps`: a list, maximum order for HGDE module
- `--depth_receptive_depth,--casual_conv_kernel_size,--scales`: number of temporal receptive aggregator layers,casual convolution kernel size, depth-wise convolution knernel size, used to config the receival field of TRA module

For all config parameter description, please refer to `./cofig.py`

## 3. Orginal Data
All datasets can be downloaded from the provided urls in `./data/orgin/`

## 4. Data preprocessing

The demo dataset can be found in `./data/`

The input data is a serialized `dict` object by `torch.save()`. It has the following keys:

- `edge_index_list`: a list of torch tensors, each of which is the edge index of single snapshot
- `pedges`, `nedges`: a list of torch tensors, each of which is the sampled positive and negative edge index of single snapshot for temporal link prediction
- `new_pedges`, `new_nedges`: a list of torch tensors, each of which is the sampled positive and negative edge index of single snapshot for temporal new link prediction
- `num_nodes`: number of nodes of the whole temporal graph
- `time_length`: number of snapshots, numerically equal to the length of `edge_index_list`, `pedges`, `nedges`, `new_pedges` and `new_nedges`
- `weights`: input node feature tensor, remain `None` if there is no input feature
