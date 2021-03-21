
import logging
import os
import glob
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from datetime import datetime


def _load_ppr(input_dir, dump_suffix, shape):
    batch_id = 0
    last_row = 0

    row_list = []
    col_list = []
    val_list = []

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
    if shape is None:
        # guess shape
        shape = (edge_rows.max() + 1, edge_cols.max() + 1)

    adj = sp.csr_matrix((edge_vals, (edge_rows, edge_cols)), shape=shape)
    concat_time = datetime.now() - concat_start
    logging.info(f"Building csr took {concat_time} seconds")
    return adj


def load_ppr(
    input_dir='/nfs/students/schmidtt/datasets/ppr/papers',
    dataset='ogbn-papers100M',
    idx=None,
    alpha=0.1,
    eps=1e-3,
    topk=64,
    ppr_normalization="row",
    split_desc=None,
    normalize=None,
    make_undirected=None,
    make_unweighted=None,
    shape=None,
):
    if input_dir is None:
        return None
    ppr_idx = None
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
    if make_unweighted is not None:
        dump_suffix += f"_unweighted{make_unweighted}"

    # check whether the precalculated ppr exists, return None if it does not.
    if len(glob.glob(str(Path(input_dir) / ("topk_ppr_" + dump_suffix)) + "*")) == 0:
        logging.info(f"No cached topk ppr found with key '{dump_suffix}' in directory '{input_dir}'")
        return None

    if split_desc is not None and idx is not None:
        ppr_idx = np.load(Path(input_dir) / f"{dump_suffix}_idx.npy")
        if len(ppr_idx) != len(idx):
            # the ppr that was precalculated with for the given configuration was calculated for a different set of nodes
            # TODO: this is only a very crude check, to make sure the idx actually matches we'd need to fully compare them, but that's expensive...
            return None

    return _load_ppr(input_dir, dump_suffix, shape)


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


# topk_train = load_ppr(input_dir='/nfs/students/schmidtt/datasets/ppr/papers100M',
#                       dataset='ogbn-papers100M',
#                       alpha=0.1,
#                       eps=1e-6,
#                       topk=256,
#                       ppr_normalization="row",
#                       split_desc="train",
#                       normalize="row",
#                       make_undirected=False,
#                       make_unweighted=True)
# print()
# print(topk_train)
# print(train_idx)
