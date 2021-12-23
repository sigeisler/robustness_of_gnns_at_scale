from typing import Union, List, Dict
import argparse
import logging
import os
import pandas as pd
import numpy as np
from rgnn_at_scale.helper.local import setup_logging, build_configs_and_run
from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.helper import ppr_utils as ppr
from rgnn_at_scale.helper.io import Storage


parser = argparse.ArgumentParser(
    description='Calculates the topk personalized page rank matrix for the given dataset & ppr configuration '
                'and adds it to the cache (disk)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--config-files', nargs='+', type=str,
                    default=[os.path.join('config', 'train', 'arxiv.yaml')],
                    help='Config YAML files. The script deduplicates the configs, but does not check them. '
                    'All other parameters will overwrite the values provided in this file')

parser.add_argument('--artifact_dir', type=str, default="cache_debug",
                    help='The folder name/path in which to look for the storage (TinyDB) objects')
parser.add_argument('--storage_type', type=str,
                    help="The name of the storage (TinyDB) table name that's supposed "
                    "to be used for caching ppr matrices")

parser.add_argument('--dataset', type=str, help='The name of the dataset for which the ppr matrix is calculated')
parser.add_argument('--data_dir', type=str, default="data/",
                    help="The folder in which the dataset can be found")
parser.add_argument('--binary_attr ', type=bool, help='Will overwrite the loaded config')
parser.add_argument('--make_undirected', type=bool, help='Will overwrite the loaded config')

parser.add_argument('--alpha', type=float,
                    help='The alpha value (restart probability, between 0 and 1) '
                    'that is used to calculate the approximate topk ppr matrix')
parser.add_argument('--eps', type=float,
                    help='The threshold used as stopping criterion for the iterative '
                    'approximation algorithm used for the ppr matrix')
parser.add_argument('--topk', type=int, help='The top k elements to keep in each row of the ppr matrix.')
parser.add_argument('--ppr_normalization', type=str,
                    help="The normalization that is applied to the top k ppr matrix before passing it "
                    "to the PPRGo model.Possible values are 'sym', 'col' and 'row' (by default 'row')")

parser.add_argument('--calc_ppr_for_all', type=bool, default=False,
                    help='whether to calculate the ppr score for all nodes (==True) '
                    'or just for the training, validation and test nodes (==False). '
                    'Generally you want to use False for training and True only when '
                    'you need the ppr matrix for a direct attack')


def calc_and_cache_ppr(dataset: str, data_dir: str, binary_attr: bool, make_undirected: bool,
                       alpha: float, eps: float, topk: int, ppr_normalization: str, calc_ppr_for_all: bool,
                       artifact_dir: str, storage_type: str):

    graph = prep_graph(dataset, "cpu",
                       dataset_root=data_dir,
                       make_undirected=make_undirected,
                       binary_attr=binary_attr,
                       return_original_split=dataset.startswith('ogbn'))

    attr, adj, labels = graph[:3]
    if graph[3] is None:
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

    def save_ppr(ppr_idx, split_desc):

        topk_ppr = ppr.topk_ppr_matrix(adj_sp, alpha, eps, ppr_idx.copy(),
                                       topk,  normalization=ppr_normalization)

        params = dict(dataset=dataset,
                      alpha=alpha,
                      ppr_idx=ppr_idx,
                      eps=eps,
                      topk=topk,
                      ppr_normalization=ppr_normalization,
                      split_desc=split_desc,
                      make_undirected=make_undirected)
        storage = Storage(artifact_dir)
        storage.save_sparse_matrix(storage_type, params,
                                   topk_ppr, ignore_duplicate=True)

    if calc_ppr_for_all:
        save_ppr(np.arange(adj_sp.shape[0]), "attack")
    else:
        save_ppr(idx_train, "train")
        save_ppr(idx_val, "val")
        save_ppr(idx_test, "test")


def maybe_get(dictionary: Dict[str, any], key: Union[str, List[str]], default=None):
    if len(key) == 1:
        key = key[-1]
    if isinstance(key, str):
        return dictionary[key] if key in dictionary.keys() else default
    return maybe_get(dictionary[key[0]], key[1:], default=default) if key[0] in dictionary.keys() else default


def main(args: argparse.Namespace):
    configs = []
    if hasattr(args, "config_files"):
        configs, run = build_configs_and_run(args.config_files, 'experiment_train.py')

        configs = [c for c in configs if "model_params" in c.keys() and (
            c["model_params"]["model"] == "PPRGo" or c["model_params"]["model"] == "RobustPPRGo")]

        configs = [dict(dataset=maybe_get(c, "dataset"),
                        data_dir=maybe_get(c, "data_dir"),
                        make_undirected=maybe_get(c, "make_undirected"),
                        binary_attr=maybe_get(c, "binary_attr"),
                        artifact_dir=maybe_get(c, ["ppr_cache_params", "data_artifact_dir"]),
                        storage_type=maybe_get(c, ["ppr_cache_params", "data_storage_type"]),
                        alpha=maybe_get(c, ["model_params", "alpha"]),
                        eps=maybe_get(c, ["model_params", "eps"]),
                        topk=maybe_get(c, ["model_params", "topk"]),
                        ppr_normalization=maybe_get(c, ["model_params", "ppr_normalization"])
                        )
                   for c in configs]

    overwrite_dict = dict(args.__dict__)
    del overwrite_dict["config_files"]
    overwrite_dict = {k: v for k, v in overwrite_dict.items() if v is not None}

    if len(configs) == 0:
        configs.append(overwrite_dict)

    for c in configs:
        c.update(overwrite_dict)

    configs = [{k: v for k, v in c.items() if v is not None} for c in configs]

    # dropping duplicates
    configs = pd.DataFrame(configs).drop_duplicates().to_dict('records')

    for config in configs:
        try:
            calc_and_cache_ppr(**config)
        except TypeError as e:
            logging.exception(e)
            continue
        except Exception as e:
            logging.exception(e)
            continue


if __name__ == '__main__':
    setup_logging()

    args = parser.parse_args()
    logging.debug(args)

    main(args)
