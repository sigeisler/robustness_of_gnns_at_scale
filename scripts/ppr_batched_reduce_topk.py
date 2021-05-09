
import seml
import logging
import os
import glob
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from datetime import datetime


from sacred import Experiment

from rgnn_at_scale.helper.local import setup_logging
from rgnn_at_scale.helper.ppr_load import load_ppr
from rgnn_at_scale.helper.io import Storage


ex = Experiment()
seml.setup_logger(ex)

setup_logging()

logging.info("start")

dataset = "ogbn-papers100M"  # "ogbn-papers100M"  # "ogbn-arxiv"
device = 0
dataset_root = "/nfs/students/schmidtt/datasets/"
input_dir = dataset_root + "ppr/papers100M/"
binary_attr = False
normalize = "row"
make_undirected = False
dir_name = '_'.join(dataset.split('-'))
#shape = [169343, 169343]
shape = [111059956, 111059956]


artifact_dir = "/nfs/students/schmidtt/cache"
model_storage_type = "ppr"
# ppr params
alpha = 0.01
eps = 1e-6
topk = 512
new_topk = 256
ppr_normalization = "row"
split_desc = "full"
logging.info(f"start {alpha} {split_desc}")


dump_suffix = f"{dataset}"
if split_desc is not None:
    dump_suffix += f"_{split_desc}"

dump_suffix += f"_alpha{int(alpha * 100)}_eps{eps:.0e}_topk{topk}"
if normalize is not None:
    dump_suffix += f"_pprnorm{ppr_normalization}_norm{normalize}"
else:
    # for backward compatibility:
    dump_suffix += f"_norm{ppr_normalization}"

if make_undirected is not None:
    dump_suffix += f"_indirect{make_undirected}"

# check whether the precalculated ppr exists, return None if it does not.
if len(glob.glob(str(Path(input_dir) / ("topk_ppr_" + dump_suffix)) + "*")) == 0:
    logging.info(f"No cached topk ppr found with key '{dump_suffix}' in directory '{input_dir}'")

batch_id = 0
last_row = 0

while True:
    start_batch = datetime.now()
    logging.info(f"Read batch {batch_id}")

    file_name = os.path.join(input_dir, f"topk_ppr_{dump_suffix}_{batch_id:08d}.npz")

    if not os.path.isfile(file_name):
        logging.info(file_name)
        break

    spi = sp.load_npz(file_name).tocoo()
    edge_rows_i, edge_cols_i, edge_vals_i = spi.row, spi.col, spi.data
    #edge_rows_i += last_row
    for row_i in np.unique(edge_rows_i):
        i_mask = edge_rows_i == row_i
        row_vals = edge_vals_i[i_mask]
        idx_sorted = np.argsort(row_vals)
        idx_topk = idx_sorted[-new_topk:]
    topk_rows = np.array(shape)

    batch_id += 1
    last_row = edge_rows_i.max() + 1

    read_time = datetime.now() - start_batch
    logging.info(f"Read {batch_id} took {read_time} seconds")
