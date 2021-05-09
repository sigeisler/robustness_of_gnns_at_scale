
import seml
import logging

from sacred import Experiment

from rgnn_at_scale.helper.local import setup_logging
from rgnn_at_scale.helper.ppr_load import load_ppr
from rgnn_at_scale.helper.io import Storage
import numpy as np

ex = Experiment()
seml.setup_logger(ex)

setup_logging()

logging.info("start")

dataset = "ogbn-papers100M"  # "ogbn-papers100M"  # "ogbn-arxiv"
device = 0
dataset_root = "data/"
ppr_input_dir = dataset_root + "ppr/papers100M/"
binary_attr = False
normalize = False
normalize_attr = False
make_undirected = False
make_unweighted = True
dir_name = '_'.join(dataset.split('-'))
#shape = [169343, 169343]
shape = [111059956, 111059956]


artifact_dir = "cache"
model_storage_type = "ppr"
# ppr params
alpha = 0.001
eps = 1e-5
topk = 128
ppr_normalization = "row"
split_desc = "attack"  # "train", "val", "test"
logging.info(f"start {alpha} {split_desc}")
topk_matrix, ppr_idx = load_ppr(input_dir=ppr_input_dir,
                                dataset=dataset,
                                alpha=alpha,
                                eps=eps,
                                topk=topk,
                                ppr_normalization=ppr_normalization,
                                split_desc="full",
                                normalize=normalize,
                                make_undirected=make_undirected,
                                make_unweighted=make_unweighted,
                                shape=shape)


logging.info("topk")
storage = Storage(artifact_dir, experiment=ex)
params = dict(dataset=dataset,
              alpha=alpha,
              ppr_idx=ppr_idx,
              eps=eps,
              topk=topk,
              ppr_normalization=ppr_normalization,
              split_desc=split_desc,
              normalize=normalize,
              normalize_attr=normalize_attr,
              make_undirected=make_undirected,
              make_unweighted=make_unweighted)
if topk_matrix is not None:
    params["ppr_idx"] = np.unique(topk_matrix.nonzero()[0])
    logging.info("trying to save sparse_matrix")
    model_path = storage.save_sparse_matrix(model_storage_type, params,
                                            topk_matrix, ignore_duplicate=True)

    logging.info("saved sparse_matrix")
    loaded_topk_train = storage.find_sparse_matrix(model_storage_type, params)

    logging.info("find_sparse_matrix")
    logging.info(loaded_topk_train.nnz)
    logging.info(loaded_topk_train.shape)

logging.info("done")
