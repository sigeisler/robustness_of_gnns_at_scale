import logging
import warnings
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
    nodes = [1854, 513, 2383]

    epsilons = [0.5, 0.75, 1]
    seed = 0
    display_steps = 10

    artifact_dir = 'cache_debug'

    model_storage_type = 'victim_cora_2'
    model_label = 'Vanilla PPRGo'

    surrogate_model_storage_type = 'nettack'
    surrogate_model_label = 'Linear GCN'

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
        surrogate_model_storage_type: str, surrogate_model_label: str, data_device: Union[str, int], display_steps: int):
    logging.info({
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'nodes': nodes, 'epsilons': epsilons,
        'binary_attr': binary_attr, 'seed': seed, 'normalize': normalize, 'normalize_attr': normalize_attr,
        'artifact_dir': artifact_dir, 'model_label': model_label, 'model_storage_type': model_storage_type,
        'device': device, "data_device": data_device, 'display_steps': display_steps
    })

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'

    results = []
    graph = prep_graph(dataset, data_device, dataset_root=data_dir,
                       normalize=normalize,
                       normalize_attr=normalize_attr,
                       make_undirected=make_undirected,
                       make_unweighted=make_unweighted,
                       binary_attr=binary_attr,
                       return_original_split=dataset.startswith('ogbn'))
    attr, adj, labels = graph[:3]
    if len(graph) == 3:
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    logging.debug("Memory Usage after loading the dataset:")
    logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    storage = Storage(artifact_dir, experiment=ex)

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr,
                        normalize=normalize,
                        normalize_attr=normalize_attr,
                        make_undirected=make_undirected,
                        make_unweighted=make_unweighted,
                        seed=seed)

    if model_label is not None and model_label:
        model_params['label'] = model_label
    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    model_params["label"] = surrogate_model_label
    surrogate_models_and_hyperparams = storage.find_models(surrogate_model_storage_type, model_params)

    assert len(surrogate_models_and_hyperparams) > 0, "No surrogate model found!"
    if len(surrogate_models_and_hyperparams) > 1:
        warnings.warn("More than one matching surrogate model found. Choose first one by default.")

    surrogate_model = surrogate_models_and_hyperparams[0][0]

    for node in nodes:
        degree = adj[node].sum() + 1

        tmp_epsilons = list(epsilons)
        tmp_epsilons.insert(0, 0)
        for eps in tmp_epsilons:
            try:
                adversary = create_attack(attack, binary_attr, attr, adj=adj, labels=labels,
                                          model=surrogate_model, idx_attack=idx_test, device=device, **attack_params)
            except Exception as e:
                logging.exception(e)
                logging.error(f"Failed to instantiate attack {attack} for model '{surrogate_model}'.")
                continue

            n_perturbations = int((eps * degree).round().item())

            # In case the model is non-deterministic to get the results either after attacking or after loading
            try:
                adversary.attack(n_perturbations, node_idx=node)
            except Exception as e:
                logging.exception(e)
                logging.error(
                    f"Failed to attack model '{surrogate_model}' using {attack} with eps {eps} at node {node}.")
                continue

            for model, hyperparams in models_and_hyperparams:
                logging.info(model)
                logging.info(hyperparams)

                adversary.set_eval_model(model)
                logits, initial_logits = adversary.evaluate_local(node)

                logging.info(
                    f'Pert. edges for node {node} and budget {n_perturbations}: {adversary.get_perturbed_edges()}')

                results.append({
                    'label': hyperparams['label'],
                    'epsilon': eps,
                    'n_perturbations': n_perturbations,
                    'degree': int(degree.item()),
                    'logits': logits.cpu(),
                    'initial_logits': initial_logits.cpu(),
                    'target': labels[node].item(),
                    'node_id': node,
                    'perturbed_edges': adversary.get_perturbed_edges().cpu().numpy()
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
