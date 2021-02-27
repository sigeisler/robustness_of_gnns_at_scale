import logging
from typing import List, Any, Dict, Sequence, Union

import seml
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sacred import Experiment
import seml
import torch
from torch.nn import functional as F
from torch_sparse import SparseTensor

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.attacks import create_attack, SPARSE_ATTACKS
from rgnn_at_scale.attacks.local_prbcd import LocalPRBCD

from rgnn_at_scale.io import Storage
from rgnn_at_scale.models import DenseGCN, GCN
from rgnn_at_scale.train import train
from rgnn_at_scale.utils import accuracy
from pprgo import utils as ppr_utils
from pprgo import ppr


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    # default params

    model_labels = ['Vanilla PPRGo']
    seeds = [0, 1, 5]
    db_collection_attacks = "kdd21_local_attack_citeseer"
    artifact_dir = 'cache'
    model_storage_type = 'attack_citeseer'
    device = "cpu"
    data_device = 'cpu'
    data_dir = './datasets'
    dataset = 'citeseer'
    make_undirected = True
    make_unweighted = True
    binary_attr = False


def flip_edges(adj: sp.csr_matrix, pert_edge_rows, pert_edge_cols):
    # not really efficient, but simple
    # and there are only a small numbers of pertubations anyway
    pert_adj = adj
    for i, j in zip(pert_edge_rows, pert_edge_cols):
        original_shape = pert_adj.shape

        adj_i = pert_adj[i].todok()
        adj_i[0, j] = 0 if adj_i[0, j] == 1 else 1
        adj_i = adj_i.tocsr()

        adj_to_i = pert_adj[:i]
        adj_from_i = pert_adj[(i + 1):]

        adj_to_i_indices, adj_to_i_indptr, adj_to_i_data = adj_to_i.indices, adj_to_i.indptr, adj_to_i.data
        adj_from_i_indices, adj_from_i_indptr, adj_from_i_data = adj_from_i.indices, adj_from_i.indptr, adj_from_i.data
        adj_i_indices, adj_i_indptr, adj_i_data = adj_i.indices, adj_i.indptr, adj_i.data

        adj_indices = np.concatenate([adj_to_i_indices, adj_i_indices, adj_from_i_indices])
        adj_data = np.concatenate([adj_to_i_data, adj_i_data, adj_from_i_data])

        ptr_i = adj_to_i_indptr[i]
        offset_from_i = ptr_i + adj_i_data.shape[0]
        adj_indptr = np.concatenate([adj_to_i_indptr[:i], np.array([ptr_i]), adj_from_i_indptr + offset_from_i])

        pert_adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr),
                                 shape=original_shape)
        test_adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr),
                                 shape=original_shape)
    return pert_adj


@ex.automain
def run(data_dir: str, dataset: str, db_collection_attacks: str, binary_attr: bool, make_undirected: bool,
        make_unweighted: bool, seeds: int, artifact_dir: str, model_labels: List[str], model_storage_type: str,
        device: Union[str, int], data_device: Union[str, int]):

    logging.info({
        'dataset': dataset, 'db_collection_attacks': db_collection_attacks, 'binary_attr': binary_attr, 'model_labels': model_labels,
        'seeds': seeds, 'artifact_dir': artifact_dir,  'model_storage_type': model_storage_type,
        'device': device, "data_device": data_device
    })

    results = []
    graph = prep_graph(dataset, data_device, dataset_root=data_dir,
                       make_undirected=make_undirected,
                       make_unweighted=make_unweighted,
                       binary_attr=binary_attr,
                       return_original_split=dataset.startswith('ogbn'))
    attr, adj, labels = graph[:3]
    if len(graph) == 3:
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

    storage = Storage(artifact_dir, experiment=ex)

    df_attack_experiments = seml.get_results(db_collection_attacks,
                                             to_data_frame=True,
                                             fields=['batch_id', 'slurm', 'config', 'result'])

    df_atk_results = [
        pd.DataFrame(r)
        for r in df_attack_experiments['result.results']
    ]

    for df_atk_result, (_, df_attack_experiment) in zip(df_atk_results, df_attack_experiments.iterrows()):
        df_atk_result['dataset'] = df_attack_experiment['config.dataset']
        df_atk_result['attack'] = df_attack_experiment['config.attack']
        df_atk_result['seed'] = df_attack_experiment['config.seed']
        df_atk_result['batch_id'] = df_attack_experiment['batch_id']

    df_atk_results = pd.concat(df_atk_results, ignore_index=True)
    df_atk_results = df_atk_results.sort_values('batch_id')
    df_atk_results = df_atk_results[df_atk_results["dataset"] == dataset]
    for model_label in model_labels:
        df_model_atk_by_model = df_atk_results[df_atk_results["label"] == model_label]

        for seed in seeds:
            df_model_atk_results = df_model_atk_by_model[df_model_atk_by_model["seed"] == seed]
            model_params = dict(dataset=dataset,
                                binary_attr=binary_attr,
                                label=model_label,
                                seed=seed)

            models_and_hyperparams = storage.find_models(model_storage_type, model_params)

            # In case the model is non-deterministic to get the results either after attacking or after loading
            torch.manual_seed(seed)
            np.random.seed(seed)

            for model, hyperparams in models_and_hyperparams:
                logging.info(model)
                logging.info(hyperparams)
                model = model.to(device)
                model.eval()
                for _, df_model_atk_result in df_model_atk_results.iterrows():
                    atk_node_ix = df_model_atk_result["node_id"]
                    perturbed_edges = df_model_atk_result["perturbed_edges"]
                    n_perturbations = df_model_atk_result["n_perturbations"]
                    if len(perturbed_edges) == 2:  # if unsucessfull, the perturbed_edges might be empty
                        eps = df_model_atk_result["epsilon"]
                        logging.info(
                            f"Recalculating logits and margin for node {atk_node_ix}, epsilon {eps} and seed {seed}")
                        pert_rows, pert_cols = np.array(perturbed_edges[0]), np.array(perturbed_edges[1])
                        if make_undirected:
                            pert_rows = np.concatenate([pert_rows, pert_cols])
                            pert_cols = np.concatenate([pert_cols, np.array(perturbed_edges[0])])

                        if isinstance(adj, SparseTensor):
                            adj = adj.to_scipy(layout="csr")

                        # TODO: we could remove this again because we have the initial logits in df_model_atk_result
                        # but it's also a nice sanity check to make sure everything works and is specified correctly
                        topk = model.topk + n_perturbations
                        ppr_topk_atk_node = ppr.topk_ppr_matrix(adj, model.alpha, model.eps, np.array(
                            [atk_node_ix]), topk, normalization=model.ppr_normalization)
                        ppr_topk_atk_node = SparseTensor.from_scipy(ppr_topk_atk_node)
                        initial_logits = F.log_softmax(model.forward(
                            attr, None, ppr_scores=ppr_topk_atk_node), dim=-1).cpu()
                        initial_classification_stats = LocalPRBCD.classification_statistics(initial_logits,
                                                                                            labels[atk_node_ix].cpu())

                        # perturbe adjacency
                        pert_adj = flip_edges(adj, pert_rows, pert_cols)

                        ppr_topk_atk_node_pert = ppr.topk_ppr_matrix(pert_adj, model.alpha, model.eps, np.array(
                            [atk_node_ix]), topk, normalization=model.ppr_normalization)

                        ppr_topk_atk_node_pert = SparseTensor.from_scipy(ppr_topk_atk_node_pert)
                        pert_logits = F.log_softmax(model.forward(
                            attr, None, ppr_scores=ppr_topk_atk_node_pert), dim=-1).cpu()
                        pert_classification_stats = LocalPRBCD.classification_statistics(pert_logits,
                                                                                         labels[atk_node_ix].cpu())

                        results.append({
                            'label': model_label,
                            'seed': seed,
                            'attack': df_model_atk_result["attack"],
                            'dataset': df_model_atk_result["dataset"],
                            'epsilon': eps,
                            'n_perturbations': n_perturbations,
                            'degree': df_model_atk_result["degree"],
                            'logits': pert_logits.cpu(),
                            'initial_logits': initial_logits.cpu(),
                            'larget': labels[atk_node_ix].item(),
                            'node_id': atk_node_ix,
                            'perturbed_edges': perturbed_edges,
                            'initial_margin_diff': df_model_atk_result["initial_margin"] - initial_classification_stats["margin"],
                            'margin_diff': df_model_atk_result["margin"] - pert_classification_stats["margin"]
                        })
                        results[-1].update(pert_classification_stats)
                        results[-1].update({
                            f'initial_{key}': value
                            for key, value
                            in initial_classification_stats.items()
                        })

                        # undo pertubation so we have a clean adjacency for the next iteration
                        #adj = flip_edges(adj, pert_rows, pert_cols)

    assert len(results) > 0

    return {
        'results': results
    }
