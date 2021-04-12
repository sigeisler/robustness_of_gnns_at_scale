
import seml
import logging

import numpy as np

from sacred import Experiment
from rgnn_at_scale.helper import ppr_utils as ppr
from rgnn_at_scale.helper.local import setup_logging
from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.helper import utils
from rgnn_at_scale.helper.io import Storage

setup_logging()

logging.info("start")

# whether to calculate the ppr score for all nodes (==True)
# or just for the training, validation and test nodes (==False)
calc_ppr_for_all = True


artifact_dir = "/nfs/students/schmidtt/cache"
model_storage_type = "ppr"

# dataset params
dataset = "ogbn-papers100M"  # "ogbn-papers100M"  # "ogbn-arxiv"
device = 0
dataset_root = "/nfs/students/schmidtt/datasets/"
binary_attr = False
normalize = "row"
make_undirected = False
make_unweighted = True


# ppr params
alpha = 0.01
eps = 1e-6
topk = 512
ppr_normalization = "row"
alpha_suffix = int(alpha * 100)

graph = prep_graph(dataset, "cpu",
                   dataset_root=dataset_root,
                   make_undirected=make_undirected,
                   make_unweighted=make_unweighted,
                   normalize=normalize,
                   binary_attr=binary_attr,
                   return_original_split=dataset.startswith('ogbn'))

attr, adj, labels = graph[:3]
if len(graph) == 3:
    idx_train, idx_val, idx_test = split(labels.cpu().numpy())
else:
    idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

logging.info("successfully read dataset")

attr, adj, labels = graph[:3]
num_nodes = attr.shape[0]
train_num_nodes = len(idx_train)
val_num_nodes = len(idx_val)
test_num_nodes = len(idx_test)
logging.info(f"Dataset has {num_nodes} nodes")
logging.info(f"Train split has {train_num_nodes} nodes")
logging.info(f"Val split has {val_num_nodes} nodes")
logging.info(f"Test split has {test_num_nodes} nodes")

ex = Experiment()
seml.setup_logger(ex)


def _save_ppr_topk(artifact_dir,
                   model_storage_type,
                   adj_sp,
                   ppr_idx,
                   alpha,
                   eps,
                   topk,
                   ppr_normalization,
                   make_undirected,
                   make_unweighted,
                   normalize,
                   split_desc):
    dump_suffix = f"{dataset}_{split_desc}_alpha{alpha_suffix}_eps{eps:.0e}_topk{topk}_pprnorm{ppr_normalization}_norm{normalize}_indirect{make_undirected}_unweighted{make_unweighted}"
    logging.info(dump_suffix)

    params = dict(dataset=dataset,
                  alpha=alpha,
                  ppr_idx=ppr_idx,
                  eps=eps,
                  topk=topk,
                  ppr_normalization=ppr_normalization,
                  split_desc=split_desc,
                  normalize=normalize,
                  make_undirected=make_undirected,
                  make_unweighted=make_unweighted)

    logging.info(f'Memory before calculating ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')
    topk_ppr = ppr.topk_ppr_matrix(adj_sp, alpha, eps, ppr_idx,
                                   topk,  normalization=ppr_normalization)

    logging.info(f'Memory after calculating ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')

    storage = Storage(artifact_dir, experiment=ex)
    storage.save_sparse_matrix(model_storage_type, params,
                               topk_ppr, ignore_duplicate=True)

    logging.info(f'Memory after saving ppr: {utils.get_max_memory_bytes() / (1024 ** 3)}')


def save_ppr_topk(artifact_dir,
                  model_storage_type,
                  adj_sp,
                  alpha,
                  eps,
                  topk,
                  ppr_normalization,
                  make_undirected,
                  make_unweighted,
                  normalize,
                  calc_ppr_for_all,
                  idx_train, idx_val, idx_test):

    def save_ppr(ppr_idx, split_desc):
        _save_ppr_topk(artifact_dir,
                       model_storage_type,
                       adj,
                       ppr_idx,
                       alpha,
                       eps,
                       topk,
                       ppr_normalization,
                       make_undirected,
                       make_unweighted,
                       normalize,
                       split_desc)

    if calc_ppr_for_all:
        save_ppr(np.arange(adj_sp.shape[0]), "full")
    else:
        save_ppr(idx_train, "train")
        save_ppr(idx_val, "val")
        save_ppr(idx_test, "test")


save_ppr_topk(artifact_dir,
              model_storage_type,
              adj,
              alpha,
              eps,
              topk,
              ppr_normalization,
              make_undirected,
              make_unweighted,
              normalize,
              calc_ppr_for_all,
              idx_train, idx_val, idx_test
              )
