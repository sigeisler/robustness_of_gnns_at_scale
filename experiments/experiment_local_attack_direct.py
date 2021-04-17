import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
from sacred import Experiment
import seml
import torch

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.attacks import create_attack, SPARSE_ATTACKS
from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.models import DenseGCN, GCN
from rgnn_at_scale.train import train
from rgnn_at_scale.helper.utils import accuracy

from rgnn_at_scale.helper import utils

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
    dataset = 'cora_ml'  # Options are 'cora_ml' and 'citeseer' (or with a big GPU 'pubmed')
    attack = 'Nettack'
    attack_params = {
        "lr_factor": 0.05,
        "search_space_size": 10000,
        "ppr_recalc_at_end": False,
    }
    # nodes based on pprgo confidence values:
    # nodes = [1262, 1352, 1275,  365,   10, 1334, 1306, 1252, 1360, 1299,  # best confidence
    #          2454, 1795, 1796,  732,  971, 2212,  245, 2365, 2789, 1407,  # worst confidence
    #          853, 1964,  630,  980,   29, 1193, 2626,  786,  402, 2515,  # random
    #          1785, 325, 2352,  668,   65, 1169, 2430,  568, 2052, 1796]  # random

    # nodes based on linear gcn confidence values
    # nodes = [333, 1854, 513,   51, 2383,  890, 1138,  216, 1079,  890,   # random
    #          1346, 186,  210,  410,  261, 1278, 2746,  113, 2579, 1192,  # random
    #          2259, 1787, 2802,  264, 1933,  580,  466, 1063,  699, 1159,  # best confidence
    #          798, 1560, 1822, 2555, 1449, 1879, 2202, 2352, 1796,  929]  # worst confidence

    nodes = [1854, 513, 2383]

    epsilons = [0.5, 0.75, 1]
    seed = 0
    display_steps = 10

    artifact_dir = 'cache_debug'
    model_storage_type = 'victim_cora_2'
    model_label = 'Vanilla PPRGo'

    data_dir = './datasets'
    binary_attr = False
    normalize = False
    normalize_attr = False
    make_undirected = True
    make_unweighted = True

    data_device = 'cpu'
    device = "cpu"


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], nodes: str, epsilons: Sequence[float],
        binary_attr: bool, make_undirected: bool, make_unweighted: bool, seed: int, normalize: bool, normalize_attr: str,
        artifact_dir: str, model_label: str, model_storage_type: str, device: Union[str, int],
        data_device: Union[str, int], display_steps: int):

    results = []
    surrogate_model_label = False

    (attr, adj, labels,
     idx_train,
     idx_val,
     idx_test,
     storage,
     _,
     model_params, _) = prepare_attack_experiment(data_dir, dataset, attack, attack_params,
                                                  epsilons, binary_attr, make_undirected,
                                                  make_unweighted,  normalize, normalize_attr, seed,
                                                  artifact_dir, pert_adj_storage_type, pert_attr_storage_type,
                                                  model_label, model_storage_type, device,
                                                  surrogate_model_label, data_device, ex)

    if model_label is not None and model_label:
        model_params['label'] = model_label
    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    torch.manual_seed(seed)
    np.random.seed(seed)

    for model, hyperparams in models_and_hyperparams:
        logging.info(model)
        logging.info(hyperparams)

        try:
            adversary = create_attack(attack, binary_attr, attr, adj=adj, labels=labels,
                                      model=model, idx_attack=idx_test, device=device, **attack_params)
        except Exception as e:
            logging.exception(e)
            logging.error(f"Failed to instantiate attack {attack} for model '{model}'.")
            continue

        for node in nodes:
            degree = adj[node].sum() + 1
            for eps in epsilons:
                n_perturbations = int((eps * degree).round().item())
                if n_perturbations == 0:
                    continue

                # In case the model is non-deterministic to get the results either after attacking or after loading
                try:
                    adversary.attack(node, n_perturbations)
                    logits, initial_logits = adversary.evaluate_local(node)
                except Exception as e:
                    logging.exception(e)
                    logging.error(
                        f"Failed to attack model '{model}' using {attack} with eps {eps} at node {node}.")
                    continue

                logging.info(
                    f'Pert. edges for node {node} and budget {n_perturbations}: {adversary.perturbed_edges}')

                results.append({
                    'label': hyperparams['label'],
                    'epsilon': eps,
                    'n_perturbations': n_perturbations,
                    'degree': int(degree.item()),
                    'logits': logits.cpu(),
                    'initial_logits': initial_logits.cpu(),
                    'larget': labels[node].item(),
                    'node_id': node,
                    'perturbed_edges': adversary.perturbed_edges.cpu().numpy()
                })

                results[-1].update(adversary.classification_statistics(logits.cpu(), labels[node].long().cpu()))
                results[-1].update({
                    f'initial_{key}': value
                    for key, value
                    in adversary.classification_statistics(initial_logits.cpu(), labels[node].long().cpu()).items()
                })
                # if hasattr(adversary, 'attack_statistics'):
                #     results[-1]['attack_statistics'] = adversary.attack_statistics

    assert len(results) > 0

    return {
        'results': results
    }
