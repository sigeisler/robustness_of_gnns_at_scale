
import seml
import logging

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
ppr_input_dir = dataset_root + "ppr/papers100M/"
binary_attr = False
normalize = "row"
make_undirected = False
make_unweighted = True
dir_name = '_'.join(dataset.split('-'))
#shape = [169343, 169343]
shape = [111059956, 111059956]


artifact_dir = "/nfs/students/schmidtt/cache"
model_storage_type = "ppr"
# ppr params
alpha = 0.1
eps = 1e-6
topk = 512
ppr_normalization = "row"
for alpha in [0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3]:
    for split_desc in ["train", "val", "test"]:
        logging.info(f"start {alpha} {split_desc}")
        topk_matrix, ppr_idx = load_ppr(input_dir=ppr_input_dir,
                                        dataset=dataset,
                                        alpha=alpha,
                                        eps=eps,
                                        topk=topk,
                                        ppr_normalization=ppr_normalization,
                                        split_desc=split_desc,
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
                      make_undirected=make_undirected,
                      make_unweighted=make_unweighted)
        if topk_matrix is not None:
            model_path = storage.save_sparse_matrix(model_storage_type, params,
                                                    topk_matrix, ignore_duplicate=True)

            logging.info("save_sparse_matrix")
            loaded_topk_train = storage.find_sparse_matrix(model_storage_type, params)

            logging.info("find_sparse_matrix")
