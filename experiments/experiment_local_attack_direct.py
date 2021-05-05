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
from experiments.common import prepare_attack_experiment, get_local_attack_nodes

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
    attack = 'LocalBatchedPRBCD'
    attack_params = {"ppr_cache_params": {
        "data_artifact_dir": "cache",
        "data_storage_type": "ppr"
    }
    }
    nodes = None  # [1854, 513, 2383]
    nodes_topk = 3

    epsilons = [0.5, 0.75, 1]
    seed = 0
    display_steps = 10

    artifact_dir = "cache"
    model_storage_type = 'victim_cora'
    model_label = 'Vanilla PPRGo'

    data_dir = './data'
    binary_attr = False
    normalize = False
    normalize_attr = False
    make_undirected = True
    make_unweighted = True

    data_device = 'cpu'
    device = "cpu"


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], nodes: str, nodes_topk: int, epsilons: Sequence[float],
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
     attack_params,
     _,
     model_params, _) = prepare_attack_experiment(data_dir, dataset, attack, attack_params,
                                                  epsilons, binary_attr, make_undirected,
                                                  make_unweighted,  normalize, normalize_attr, seed,
                                                  artifact_dir, None, None,
                                                  model_label, model_storage_type, device,
                                                  surrogate_model_label, data_device, ex)

    if model_label is not None and model_label:
        model_params['label'] = model_label
    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    for model, hyperparams in models_and_hyperparams:
        model.to(device)
        model_label = hyperparams["label"]

        try:
            adversary = create_attack(attack, binary_attr, attr, adj=adj, labels=labels,
                                      model=model, idx_attack=idx_test, device=device, data_device=data_device, **attack_params)
        except Exception as e:
            logging.exception(e)
            logging.error(f"Failed to instantiate attack {attack} for model '{model_label}'.")
            continue

        tmp_nodes = np.array(nodes)
        if nodes is None:
            tmp_nodes = get_local_attack_nodes(adversary, binary_attr, attr, adj, labels,
                                               model, idx_test, device, attack_params, topk=nodes_topk)
        tmp_nodes = [int(i) for i in tmp_nodes]

        for node in tmp_nodes:
            degree = adj[node].sum() + 1
            for eps in epsilons:
                n_perturbations = int((eps * degree).round().item())
                if n_perturbations == 0:
                    continue

                # In case the model is non-deterministic to get the results either after attacking or after loading
                try:
                    adversary.attack(n_perturbations, node_idx=node)
                except Exception as e:
                    logging.exception(e)
                    logging.error(
                        f"Failed to attack model '{model_label}' using {attack} with eps {eps} at node {node}.")
                    continue

                logits, initial_logits = adversary.evaluate_local(node)

                logging.info(
                    f'Evaluated model {model_label} using {attack} with pert. edges for node {node} and budget {n_perturbations}: {adversary.get_perturbed_edges()}')

                results.append({
                    'label': model_label,
                    'epsilon': eps,
                    'n_perturbations': n_perturbations,
                    'degree': int(degree.item()),
                    'logits': logits.cpu(),
                    'initial_logits': initial_logits.cpu(),
                    'larget': labels[node].item(),
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
        del adversary
    assert len(results) > 0

    return {
        'results': results
    }
