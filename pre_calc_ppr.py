
import logging
from pathlib import Path

import os.path as osp
from typing import Any, Dict, Union

import numpy as np
import math
from sacred import Experiment
import seml
import torch
import torch_sparse
import scipy.sparse as sp

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.io import Storage
from rgnn_at_scale.models import create_model
from rgnn_at_scale.train import train
from rgnn_at_scale.utils import accuracy
from pprgo import utils as ppr_utils
from pprgo import ppr

from torch_geometric.utils import add_remaining_self_loops, remove_isolated_nodes


from ogb.nodeproppred import PygNodePropPredDataset
from rgnn_at_scale.local import setup_logging

setup_logging()

logging.info("start")
dataset = "ogbn-arxiv"  # "ogbn-papers100M"
device = "cpu"
dataset_root = "/nfs/students/schmidtt/datasets/"
binary_attr = False
topk_batch_size = 10240
dir_name = '_'.join(dataset.split('-'))

# check if previously-downloaded folder exists.
# If so, use that one.
if osp.exists(osp.join(dataset_root, dir_name + '_pyg')):
    dir_name = dir_name + '_pyg'
dataset_dir = dataset_root + dir_name

# ppr params
alpha = 0.5
eps = 1e-2
topk = 32
ppr_normalization = "row"
alpha_suffix = int(alpha * 100)
dump_suffix = f"alpha{alpha_suffix}_eps{eps:.0e}_topk{topk}_norm{ppr_normalization}"

logging.info("Start")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

pyg_dataset = PygNodePropPredDataset(root=dataset_root, name=dataset)
logging.info("Loaded PygNodePropPredDataset into memory.")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

data = pyg_dataset[0]
logging.info("data in memory.")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

if hasattr(data, '__num_nodes__'):
    num_nodes = data.__num_nodes__
else:
    num_nodes = data.num_nodes

if hasattr(pyg_dataset, 'get_idx_split'):
    split = pyg_dataset.get_idx_split()
else:
    split = dict(
        train=data.train_mask.nonzero().squeeze(),
        valid=data.val_mask.nonzero().squeeze(),
        test=data.test_mask.nonzero().squeeze()
    )

# converting to numpy arrays, so we don't have to handle different
# array types (tensor/numpy/list) later on.
# Also we need numpy arrays because Numba cant determine type of torch.Tensor
split = {k: v.numpy() for k, v in split.items()}

logging.info("Dataset split loaded.")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

edge_index, edge_weight = add_remaining_self_loops(
    data.edge_index.to(device),
    torch.ones(data.edge_index.size(1), device=device).float(),
    num_nodes=num_nodes
)
logging.info("Added self loops")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

attr = data.x.to(device)
logging.info("Load attributes")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

labels = data.y.squeeze().to(device)
logging.info("Load labels")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

idx_train, idx_val, idx_test = split['train'], split['valid'], split['test']

num_nodes, _ = attr.shape
logging.info("Load ppr_idx")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

adj_sp = sp.csr_matrix(sp.coo_matrix((edge_weight, edge_index),
                                     (num_nodes, num_nodes)))

logging.info("Load adj_sp")
logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))


ppr_idx = np.arange(num_nodes)
num_batches = math.ceil(len(ppr_idx) / topk_batch_size)

for i in range(num_batches):
    logging.info(f"topk for batch {i+1} of {num_batches}")
    batch_idx = ppr_idx[(i * topk_batch_size):(i + 1) * topk_batch_size]
    idx_size = len(batch_idx)
    logging.info(f"batch has {idx_size} elements.")
    topk_ppr = ppr.topk_ppr_matrix(adj_sp, alpha, eps, batch_idx,
                                   topk,  normalization=ppr_normalization)
    logging.info("Load topk_ppr")
    logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

    sp.save_npz(dataset_dir + f"/topk_ppr_{dump_suffix}_{i:08d}.npz", topk_ppr)
