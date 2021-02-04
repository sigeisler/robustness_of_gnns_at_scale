
import logging
import os

import numpy as np
import scipy.sparse as sp
from datetime import datetime


def load_ppr(
    input_dir='/nfs/students/schmidtt/datasets/ppr/papers',
    dataset='ogbn-papers100M',
    alpha=0.1,
    eps=1e-3,
    topk=64,
    ppr_normalization="row"
):
    batch_id = 0
    last_row = 0

    row_list = []
    col_list = []
    val_list = []

    while True:
        start_batch = datetime.now()
        logging.info(f"Read batch {batch_id}")

        dump_suffix = f"{dataset}_alpha{int(alpha * 100)}_eps{eps:.0e}_topk{topk}_norm{ppr_normalization}"
        file_name = os.path.join(input_dir, f"topk_ppr_{dump_suffix}_{batch_id:08d}.npz")

        if not os.path.isfile(file_name):
            logging.info(file_name)
            break

        spi = sp.load_npz(file_name).tocoo()
        edge_rows_i, edge_cols_i, edge_vals_i = spi.row, spi.col, spi.data
        edge_rows_i += last_row

        row_list.append(edge_rows_i)
        col_list.append(edge_cols_i)
        val_list.append(edge_vals_i)

        batch_id += 1
        last_row = edge_rows_i.max() + 1

        read_time = datetime.now() - start_batch
        logging.info(f"Read {batch_id} took {read_time} seconds")

    logging.info("Start concatinating")
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

    concat_start = datetime.now()
    adj = sp.csr_matrix((edge_vals, (edge_rows, edge_cols)), shape=(edge_rows.max() + 1, edge_cols.max() + 1))
    concat_time = datetime.now() - concat_start
    logging.info(f"Building csr took {concat_time} seconds")
    return adj


def load_ppr_csr(
    input_dir='/nfs/students/schmidtt/datasets/ppr/papers100M',
    dataset='ogbn-papers100M',
    alpha=0.1,
    eps=1e-3,
    topk=64,
    ppr_normalization="row"
):
    batch_id = 0

    csrs = []

    while True:
        start_batch = datetime.now()
        logging.info(f"Read batch {batch_id}")

        dump_suffix = f"{dataset}_alpha{int(alpha * 100)}_eps{eps:.0e}_topk{topk}_norm{ppr_normalization}"
        file_name = os.path.join(input_dir, f"topk_ppr_{dump_suffix}_{batch_id:08d}.npz")

        if not os.path.isfile(file_name):
            logging.info(file_name)
            break

        csrs.append(sp.load_npz(file_name))

        batch_id += 1

        read_time = datetime.now() - start_batch
        logging.info(f"Read {batch_id} took {read_time} seconds")

    concat_start = datetime.now()
    adj = sp.vstack(csrs)
    concat_time = datetime.now() - concat_start
    logging.info(f"Building csr took {concat_time} seconds")
    return adj
