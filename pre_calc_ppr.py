
import logging

import math
import os.path as osp
from pathlib import Path
import numba
import numpy as np
import scipy.sparse as sp
import psutil

from pprgo import utils as ppr_utils
from pprgo import ppr


from rgnn_at_scale.local import setup_logging
from rgnn_at_scale.data import prep_graph, split

setup_logging()

logging.info("start")

logging.info("start")
dataset = "ogbn-arxiv"  # "ogbn-papers100M"  # "ogbn-arxiv"
device = "cpu"
dataset_root = "/nfs/students/schmidtt/datasets/"
output_dir = dataset_root + "ppr/"
binary_attr = False
topk_batch_size = 10240
dir_name = '_'.join(dataset.split('-'))

# dataset = "ogbn-papers100M"  # "ogbn-arxiv"  # "ogbn-papers100M"  # "ogbn-arxiv"
# device = 0
# dataset_root = "/nfs/students/schmidtt/datasets/"
# output_dir = dataset_root + "ppr/papers/"
# binary_attr = False
# topk_batch_size = int(1e6)
# dir_name = '_'.join(dataset.split('-'))


# ppr params
alpha = 0.1
eps = 1e-4
topk = 64
ppr_normalization = "row"
alpha_suffix = int(alpha * 100)

graph = prep_graph(dataset, "cpu", dataset_root=dataset_root,
                   make_undirected=False,
                   make_unweighted=True,
                   binary_attr=False,
                   return_original_split=dataset.startswith('ogbn'))

logging.info("successfully read dataset")

attr, adj, labels = graph[:3]
num_nodes = attr.shape[0]
logging.info(f"Dataset has {num_nodes} nodes")


def save_ppr_topk(topk_batch_size,
                  output_dir,
                  adj_sp,
                  num_nodes,
                  alpha,
                  eps,
                  topk,
                  ppr_normalization):
    dump_suffix = f"{dataset}_alpha{alpha_suffix}_eps{eps:.0e}_topk{topk}_norm{ppr_normalization}"
    ppr_idx = np.arange(num_nodes)
    num_batches = math.ceil(num_nodes / topk_batch_size)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(num_batches):
        logging.info(f"topk for batch {i+1} of {num_batches}")

        batch_idx = ppr_idx[(i * topk_batch_size):(i + 1) * topk_batch_size]
        idx_size = len(batch_idx)
        logging.info(f"batch has {idx_size} elements.")
        topk_ppr = ppr.topk_ppr_matrix(adj_sp, alpha, eps, batch_idx,
                                       topk,  normalization=ppr_normalization)
        logging.info("Load topk_ppr")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))
        file_name = f"topk_ppr_{dump_suffix}_{i:08d}.npz"

        sp.save_npz(output_dir + file_name, topk_ppr)


save_ppr_topk(topk_batch_size, output_dir, adj, num_nodes, alpha, eps, topk, ppr_normalization)
