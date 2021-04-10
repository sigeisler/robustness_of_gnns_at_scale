
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
    binary_attr = False
    normalize = False
    make_undirected = True
    make_unweighted = True
    normalize_attr = False
    seed = 0

    attack = 'GreedyRBCD'
    attack_params = {}
    epsilons = [0.01]

    artifact_dir = 'cache_debug'
    model_storage_type = 'victim_cora_2'
    pert_adj_storage_type = 'evasion_attack_adj'
    pert_attr_storage_type = 'evasion_attack_attr'
    model_label = "Vanilla GDC"

    device = "cpu"
    data_device = "cpu"

    display_steps = 10


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float], binary_attr: bool,
        make_undirected: bool, make_unweighted: bool,  normalize: bool, normalize_attr: str,  seed: int,
        artifact_dir: str, pert_adj_storage_type: str, pert_attr_storage_type: str, model_label: str, model_storage_type: str,
        device: Union[str, int], data_device: Union[str, int], display_steps: int):
    logging.info({
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'epsilons': epsilons,
        'make_undirected': make_undirected, 'make_unweighted': make_unweighted, 'normalize': normalize,
        'normalize_attr': normalize_attr, 'binary_attr': binary_attr,
        'seed': seed, 'artifact_dir': artifact_dir, 'pert_adj_storage_type': pert_adj_storage_type,
        'pert_attr_storage_type': pert_attr_storage_type, 'model_label': model_label,
        'model_storage_type': model_storage_type, 'device': device, 'display_steps': display_steps
    })

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps > 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'
    assert model_label is not None, "Model label must not be None"

    # To increase consistency between runs
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    storage = Storage(artifact_dir, experiment=ex)

    pert_params = dict(dataset=dataset,
                       binary_attr=binary_attr,
                       normalize=normalize,
                       normalize_attr=normalize_attr,
                       make_undirected=make_undirected,
                       make_unweighted=make_unweighted,
                       seed=seed,
                       attack=attack,
                       model=model_label,
                       surrogate_model=False,
                       attack_params=attack_params)

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr,
                        normalize=normalize,
                        normalize_attr=normalize_attr,
                        make_undirected=make_undirected,
                        make_unweighted=make_unweighted,
                        label=model_label,
                        seed=seed)

    if make_undirected:
        m = adj.nnz() / 2
    else:
        m = adj.nnz()

    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    for model, hyperparams in models_and_hyperparams:
        model_label = hyperparams["label"]
        try:
            adversary = create_attack(attack, binary_attr, attr, adj=adj, labels=labels,
                                      model=model, idx_attack=idx_test, device=device, **attack_params)
        except Exception as e:
            logging.exception(e)
            logging.error(f"Failed to instantiate attack {attack} for model '{model_label}'.")
            continue

        for epsilon in epsilons:
            n_perturbations = int(round(epsilon * m))

            pert_adj = storage.load_artifact(pert_adj_storage_type, {**pert_params, **{'epsilon': epsilon}})
            pert_attr = storage.load_artifact(pert_attr_storage_type, {**pert_params, **{'epsilon': epsilon}})

            if pert_adj is not None and pert_attr is not None:
                logging.info(
                    f"Found cached perturbed adjacency and attribute matrix for model '{model_label}' and eps {epsilon}")
                adversary.set_pertubations(pert_adj, pert_attr)
            else:
                logging.info(
                    f"No cached perturbations found for model '{model_label}' and eps {epsilon}. Execute attack...")
                adversary.attack(n_perturbations)
                pert_adj, pert_attr = adversary.get_pertubations()

                storage.save_artifact(pert_adj_storage_type, {**pert_params, **{'epsilon': epsilon}}, pert_adj)
                storage.save_artifact(pert_attr_storage_type, {**pert_params, **{'epsilon': epsilon}}, pert_attr)

            logits, accuracy = adversary.evaluate_global(idx_test)

            results.append({
                'label': model_label,
                'epsilon': epsilon,
                'accuracy': accuracy
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # evaluate clean accuracy
        adversary.set_pertubations(adj, attr)

        logits, accuracy = adversary.evaluate_global(idx_test)
        results.append({
            'label': model_label,
            'epsilon': 0,
            'accuracy': accuracy
        })

    assert len(results) > 0

    return {
        'results': results
    }
