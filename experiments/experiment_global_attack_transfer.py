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
from experiments.common import (load_perturbed_data_if_exists, train_surrogate_model,
                                run_attacks, evaluate_global_attack)

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
    data_dir = './datasets'
    attack = 'PRBCD'
    attack_params = {}
    epsilons = [0, 0.1, 0.25]
    surrogate_params = {
        'n_filters': 64,
        'dropout': 0.5,
        'train_params': {
            'lr': 1e-2,
            'weight_decay': 1e-3,  # TODO: 5e-4,
            'patience': 100,
            'max_epochs': 3000
        }
    }
    binary_attr = False
    normalize = False
    make_undirected = True
    make_unweighted = True
    normalize_attr = None
    seed = 0
    artifact_dir = 'cache'
    pert_adj_storage_type = 'evasion_transfer_attack_adj'
    pert_attr_storage_type = 'evasion_transfer_attack_attr'
    model_storage_type = 'attack_cora'
    device = "cpu"
    data_device = "cpu"
    display_steps = 10
    model_label = None


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float], binary_attr: bool,
        make_undirected: bool, make_unweighted: bool,  normalize: bool, normalize_attr: str, surrogate_params: Dict[str, Any], seed: int,
        artifact_dir: str, pert_adj_storage_type: str, pert_attr_storage_type: str, model_label: str, model_storage_type: str,
        device: Union[str, int], data_device: Union[str, int], display_steps: int):
    logging.info({
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'epsilons': epsilons,
        'make_undirected': make_undirected, 'make_unweighted': make_unweighted, 'normalize': normalize,
        'normalize_attr': normalize_attr, 'binary_attr': binary_attr, 'surrogate_params': surrogate_params,
        'seed': seed, 'artifact_dir': artifact_dir, 'pert_adj_storage_type': pert_adj_storage_type,
        'pert_attr_storage_type': pert_attr_storage_type, 'model_label': model_label,
        'model_storage_type': model_storage_type, 'device': device, 'display_steps': display_steps
    })

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'
    assert 'train_params' in surrogate_params, '`surrogate` must contain the field `train_params`'

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
    n_features = attr.shape[1]
    n_classes = int(labels.max() + 1)

    params = dict(dataset=dataset,
                  binary_attr=binary_attr,
                  normalize=normalize,
                  normalize_attr=normalize_attr,
                  make_undirected=make_undirected,
                  make_unweighted=make_unweighted,
                  seed=seed,
                  attack=attack,
                  surrogate_params=surrogate_params,
                  attack_params=attack_params)

    storage = Storage(artifact_dir, experiment=ex)

    adj_per_eps, attr_per_eps = load_perturbed_data_if_exists(
        storage, pert_adj_storage_type, pert_attr_storage_type, params, epsilons)

    if len(adj_per_eps) == 0:
        surrogate_model = train_surrogate_model(attack, adj, attr, labels, idx_train, idx_val,
                                                idx_test, n_classes, n_features, surrogate_params, display_steps, seed, device)
        adj_per_eps, attr_per_eps = run_attacks(attack, epsilons, binary_attr, attr, adj, labels, surrogate_model, idx_test, attack_params,
                                                params, storage, pert_adj_storage_type, pert_attr_storage_type, seed, device)

    adj = adj.to('cpu')
    attr = attr.to('cpu')
    if epsilons[0] == 0:
        adj_per_eps.insert(0, adj)
        attr_per_eps.insert(0, attr)

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

    results = evaluate_global_attack(models_and_hyperparams, labels, epsilons, adj_per_eps,
                                     attr_per_eps, seed, device, idx_test)

    return {
        'results': results
    }
