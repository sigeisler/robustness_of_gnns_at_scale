
import logging

import math
import numpy as np
import scipy.sparse as sp
from pathlib import Path

from rgnn_at_scale.helper import utils
from rgnn_at_scale.helper import ppr_utils as ppr


from rgnn_at_scale.helper.local import setup_logging
from rgnn_at_scale.data import prep_graph, split

setup_logging()

logging.info("start")

# whether to calculate the ppr score for all nodes (==True)
# or just for the training, validation and test nodes (==False)
calc_ppr_for_all = True

# dataset = "ogbn-arxiv"  # "ogbn-papers100M"  # "ogbn-arxiv"
# device = "cpu"
# dataset_root = "/nfs/students/schmidtt/datasets/"
# output_dir = dataset_root + "ppr/arxiv/"
# binary_attr = False
# topk_batch_size = 10240  # int(1e5)
# dir_name = '_'.join(dataset.split('-'))

dataset = "ogbn-papers100M"  # "ogbn-papers100M"  # "ogbn-arxiv"
device = 0
dataset_root = "data/"
output_dir = dataset_root + "ppr/papers100M/"
binary_attr = False
make_undirected = False
topk_batch_size = int(1e5)
dir_name = '_'.join(dataset.split('-'))


# ppr params
alpha = 0.001
eps = 1e-5
topk = 128
ppr_normalization = "row"
alpha_suffix = int(alpha * 100)

graph = prep_graph(dataset, "cpu",
                   dataset_root=dataset_root,
                   make_undirected=make_undirected,
                   binary_attr=binary_attr,
                   return_original_split=dataset.startswith('ogbn'))

attr, adj, labels = graph[:3]
if len(graph) == 3:
    idx_train, idx_val, idx_test = split(labels.cpu().numpy())
else:
    idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

logging.info("successfully read dataset")

attr, adj, labels = graph[:3]
adj_sp = adj.to_scipy(layout="csr")
num_nodes = attr.shape[0]
train_num_nodes = len(idx_train)
val_num_nodes = len(idx_val)
test_num_nodes = len(idx_test)
logging.info(f"Dataset has {num_nodes} nodes")
logging.info(f"Train split has {train_num_nodes} nodes")
logging.info(f"Val split has {val_num_nodes} nodes")
logging.info(f"Test split has {test_num_nodes} nodes")


def _save_ppr_topk(topk_batch_size,
                   output_dir,
                   adj_sp,
                   ppr_idx,
                   alpha,
                   eps,
                   topk,
                   ppr_normalization,
                   make_undirected,
                   split_desc,
                   batch_start_idx=0,
                   num_batches=None):
    dump_suffix = f"{dataset}_{split_desc}_alpha{alpha_suffix}_eps{eps:.0e}_topk{topk}_pprnorm{ppr_normalization}_indirect{make_undirected}"
    logging.info(dump_suffix)
    num_nodes = len(ppr_idx)
    if num_batches is None:
        num_batches = math.ceil(num_nodes / topk_batch_size)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nnz = 0
    for i in range(batch_start_idx, num_batches):
        logging.info(f"topk for batch {i+1} of {num_batches}")

        batch_idx = ppr_idx[(i * topk_batch_size):(i + 1) * topk_batch_size]
        idx_size = len(batch_idx)
        logging.info(f"batch has {idx_size} elements.")
        topk_ppr = ppr.topk_ppr_matrix(adj_sp, alpha, eps, batch_idx,
                                       topk,  normalization=ppr_normalization)
        logging.info("calculated topk_ppr")
        logging.info(utils.get_max_memory_bytes() / (1024 ** 3))
        file_name = f"topk_ppr_{dump_suffix}_{i:08d}.npz"
        nnz += topk_ppr.nnz
        sp.save_npz(output_dir + file_name, topk_ppr)
    np.save(output_dir + f"{dump_suffix}_idx.npy", ppr_idx)
    logging.info(f"nnz {nnz}")


def save_ppr_topk(topk_batch_size,
                  output_dir,
                  adj_sp,
                  alpha,
                  eps,
                  topk,
                  ppr_normalization,
                  make_undirected,
                  calc_ppr_for_all,
                  idx_train, idx_val, idx_test):

    def save_ppr(ppr_idx, split_desc):
        _save_ppr_topk(topk_batch_size,
                       output_dir,
                       adj_sp,
                       ppr_idx,
                       alpha,
                       eps,
                       topk,
                       ppr_normalization,
                       make_undirected,
                       split_desc,
                       batch_start_idx=1035)
    if calc_ppr_for_all:
        save_ppr(np.arange(adj_sp.shape[0]), "full")
    else:
        save_ppr(idx_train, "train")
        save_ppr(idx_val, "val")
        save_ppr(idx_test, "test")


save_ppr_topk(topk_batch_size,
              output_dir,
              adj_sp,
              alpha,
              eps,
              topk,
              ppr_normalization,
              make_undirected,
              calc_ppr_for_all,
              idx_train, idx_val, idx_test
              )
