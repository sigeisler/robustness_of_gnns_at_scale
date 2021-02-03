
import logging

import math
import os.path as osp
from pathlib import Path
import numba
import numpy as np
import scipy.sparse as sp
from datetime import datetime

from joblib import Parallel, delayed
import multiprocessing
import torch
from torch_sparse import SparseTensor

from pprgo import utils as ppr_utils
from pprgo import ppr


from rgnn_at_scale.local import setup_logging
from rgnn_at_scale.data import prep_graph, split

setup_logging()

logging.info("start")

# dataset = "ogbn-arxiv"  # "ogbn-papers100M"  # "ogbn-arxiv"
# dataset_root = "/nfs/students/schmidtt/datasets/"
# input_dir = dataset_root + "ppr/"
# topk_batch_size = 10240
# dir_name = '_'.join(dataset.split('-'))
# num_batches = 17
# num_nodes = int(169343)

dataset = "ogbn-papers100M"  # "ogbn-arxiv"  # "ogbn-papers100M"  # "ogbn-arxiv"
dataset_root = "/nfs/students/schmidtt/datasets/"
input_dir = dataset_root + "ppr/papers/"
binary_attr = False
topk_batch_size = int(1e6)
num_nodes = int(111059956)
num_batches = 112

# ppr params
alpha = 0.1
eps = 1e-3
topk = 64
ppr_normalization = "row"
alpha_suffix = int(alpha * 100)

start = datetime.now()


def load_ppr_topk_parallel(topk_batch_size,
                           input_dir,
                           alpha,
                           eps,
                           topk,
                           ppr_normalization,
                           batch_idx):
    dump_suffix = f"{dataset}_alpha{alpha_suffix}_eps{eps:.0e}_topk{topk}_norm{ppr_normalization}"
    partial_pprs = []
    for i in batch_idx:
        start_batch = datetime.now()
        logging.info(f"Read batch {i}")
        file_name = f"topk_ppr_{dump_suffix}_{i:08d}.npz"
        in_file = input_dir + file_name
        exists = osp.exists(in_file)
        if exists:
            partial_ppr = sp.load_npz(in_file)
            partial_pprs.append({"batch_id": i,
                                 "partial_ppr": partial_ppr})
        else:
            partial_pprs.append({"batch_id": i,
                                 "partial_ppr": None})
        read_time = datetime.now() - start_batch
        logging.info(f"Read {i} took {read_time} seconds")

    return partial_pprs


num_cores = multiprocessing.cpu_count()
num_cores = 20
logging.info(f"Read using {num_cores} cores")
num_batches_per_process = math.ceil((num_batches + 1) / num_cores)

inputs = [list(range(i * num_batches_per_process, (i + 1) * num_batches_per_process))
          for i in range(num_cores)
          if i * num_batches_per_process < num_batches]

logging.info(inputs)

# results = Parallel(n_jobs=num_cores)(delayed(load_ppr_topk_parallel)(topk_batch_size,
#                                                                      input_dir,
#                                                                      alpha,
#                                                                      eps,
#                                                                      topk,
#                                                                      ppr_normalization,
#                                                                      i) for i in inputs)

results = [load_ppr_topk_parallel(topk_batch_size,
                                  input_dir,
                                  alpha,
                                  eps,
                                  topk,
                                  ppr_normalization,
                                  list(range(num_batches)))]

logging.info("Read took time: " + str(datetime.now() - start))
results = [item for sublist in results for item in sublist]

sp_0 = results[0]["partial_ppr"].tocoo()
edge_rows, edge_cols, edge_vals = sp_0.row, sp_0.col, sp_0.data
edge_rows += results[0]["batch_id"] * topk_batch_size

row_list = [edge_rows]
col_list = [edge_cols]
val_list = [edge_vals]

for i in range(1, num_batches):
    if results[i]["partial_ppr"] is not None:
        spi = results[i]["partial_ppr"].tocoo()
        edge_rows_i, edge_cols_i, edge_vals_i = spi.row, spi.col, spi.data
        edge_rows_i += results[i]["batch_id"] * topk_batch_size

        row_list.append(edge_rows_i)
        col_list.append(edge_cols_i)
        val_list.append(edge_vals_i)


logging.info(f"Start concatinating")
concat_start = datetime.now()

edge_rows = np.concatenate(row_list)
concat_time = datetime.now() - concat_start
logging.info(f"Concat took {concat_time} seconds")

concat_start = datetime.now()
edge_cols = np.concatenate(col_list)
concat_time = datetime.now() - concat_start
logging.info(f"Concat took {concat_time} seconds")

concat_start = datetime.now()
edge_vals = np.concatenate(val_list)
concat_time = datetime.now() - concat_start
logging.info(f"Concat took {concat_time} seconds")

ppr_matrix = sp.coo_matrix((edge_vals, (edge_rows, edge_cols)), shape=(num_nodes, num_nodes))

logging.info("Read and convert took time: " + str(datetime.now() - start))
print(ppr_matrix)
