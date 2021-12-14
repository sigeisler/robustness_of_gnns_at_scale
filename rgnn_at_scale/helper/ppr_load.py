
import logging
import os
import glob
from pathlib import Path

import array
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from rgnn_at_scale.helper import utils


class IncrementalCSRMatrix(object):

    def __init__(self, shape, dtype):

        if dtype is np.dtype(np.int32):
            type_flag = 'i'
        elif dtype is np.dtype(np.int64):
            type_flag = 'l'
        elif dtype is np.dtype(np.float32):
            type_flag = 'f'
        elif dtype is np.dtype(np.float64):
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = shape

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)

    def append(self, row, col, val):

        m, n = self.shape

        for r, c, v in zip(row, col, val):
            self.rows.append(r)
            self.cols.append(c)
            self.data.append(v)

    def tocsr(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        logging.info("load rows")
        cols = np.frombuffer(self.cols, dtype=np.int32)
        logging.info("load cols")
        data = np.frombuffer(self.data, dtype=self.dtype)
        logging.info("load vals")

        return sp.csr_matrix((data, (rows, cols)),
                             shape=self.shape)

    def __len__(self):

        return len(self.data)


def _load_ppr(input_dir, dump_suffix, shape):
    batch_id = 0
    last_row = 0

    row_list = []
    col_list = []
    val_list = []
    start_read = datetime.now()
    while True:
        start_batch = datetime.now()
        logging.info(f"Read batch {batch_id}")

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
        logging.info(f'Memory when reading ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    read_time = datetime.now() - start_read
    logging.info(f"Read took {read_time} seconds")
    logging.info(f'Memory after reading ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    logging.info("Start concatinating")
    concat_start = datetime.now()
    edge_rows = np.concatenate(row_list)
    concat_time = datetime.now() - concat_start
    del row_list
    logging.info(f"Concat took {concat_time} seconds")
    logging.info(f'Memory concat row ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    concat_start = datetime.now()
    edge_cols = np.concatenate(col_list)
    concat_time = datetime.now() - concat_start
    del col_list
    logging.info(f"Concat took {concat_time} seconds")
    logging.info(f'Memory concat col ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    concat_start = datetime.now()
    edge_vals = np.concatenate(val_list)
    concat_time = datetime.now() - concat_start
    del val_list
    logging.info(f"Concat took {concat_time} seconds")
    logging.info(f'Memory concat val ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    concat_start = datetime.now()
    logging.info("try build ppr")
    ppr = sp.csr_matrix((edge_vals, (edge_rows, edge_cols)),
                        shape=shape)

    concat_time = datetime.now() - concat_start
    logging.info(f"Building csr took {concat_time} seconds")
    logging.info(f'Memory after building ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    return ppr


def load_ppr(
    input_dir='datasets/ppr/papers',
    dataset='ogbn-papers100M',
    idx=None,
    alpha=0.1,
    eps=1e-3,
    topk=64,
    ppr_normalization="row",
    split_desc=None,
    make_undirected=None,
    shape=None,
):
    if input_dir is None:
        return None, None
    dump_suffix = f"{dataset}"
    if split_desc is not None:
        dump_suffix += f"_{split_desc}"

    dump_suffix += f"_alpha{int(alpha * 100)}_eps{eps:.0e}_topk{topk}"
    dump_suffix += f"_pprnorm{ppr_normalization}"

    if make_undirected is not None:
        dump_suffix += f"_indirect{make_undirected}"

    # check whether the precalculated ppr exists, return None if it does not.
    if len(glob.glob(str(Path(input_dir) / ("topk_ppr_" + dump_suffix)) + "*")) == 0:
        logging.info(f"No cached topk ppr found with key '{dump_suffix}' in directory '{input_dir}'")
        return None, None

    ppr_idx = None
    if split_desc is not None and idx is not None:
        ppr_idx = np.load(Path(input_dir) / f"{dump_suffix}_idx.npy")

    return _load_ppr(input_dir, dump_suffix, shape), ppr_idx


def load_ppr_csr(
    input_dir='datasets/ppr/papers100M',
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
