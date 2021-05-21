import collections
import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
from sacred import Experiment
import seml

from rgnn_at_scale.attacks import create_attack
from experiments.common import prepare_attack_experiment, get_local_attack_nodes

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
    dataset = 'citeseer'  # Options are 'cora_ml' and 'citeseer' (or with a big GPU 'pubmed')
    attack = 'LocalBatchedPRBCD'
    attack_params = {
        "ppr_cache_params": {
            "data_artifact_dir": "cache",
            "data_storage_type": "ppr"
        }
    }
    nodes = None
    nodes_topk = 40

    epsilons = [0.5]  # , 0.75, 1]
    seed = 1

    artifact_dir = "cache"
    model_storage_type = 'pretrained'
    model_label = 'Vanilla PPRGo'

    data_dir = './datasets'
    binary_attr = False
    make_undirected = False

    data_device = 'cpu'
    device = "cpu"
    debug_level = "info"


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], nodes: str, nodes_topk: int, seed: int,
        epsilons: Sequence[float], binary_attr: bool, make_undirected: bool, artifact_dir: str, model_label: str,
        model_storage_type: str, device: Union[str, int], data_device: Union[str, int], debug_level: str):

    results = []
    surrogate_model_label = False

    (attr, adj, labels, _, _, idx_test, storage, attack_params, _, model_params, _) = prepare_attack_experiment(
        data_dir, dataset, attack, attack_params, epsilons, binary_attr, make_undirected, seed, artifact_dir,
        None, None, model_label, model_storage_type, device, surrogate_model_label, data_device, debug_level, ex
    )

    if model_label is not None and model_label:
        model_params['label'] = model_label
    models_and_hyperparams = storage.find_models(model_storage_type, model_params)
    logging.info(f"Found {len(models_and_hyperparams)} models with label '{model_label}' to attack.")

    for model, hyperparams in models_and_hyperparams:
        model.to(device)
        model_label = hyperparams["label"]

        try:
            adversary = create_attack(attack, attr=attr, adj=adj, labels=labels, model=model, idx_attack=idx_test,
                                      device=device, data_device=data_device, binary_attr=binary_attr,
                                      make_undirected=make_undirected, **attack_params)

            if hasattr(adversary, "ppr_matrix"):
                adversary.ppr_matrix.save_to_storage()
        except Exception as e:
            logging.exception(e)
            logging.error(f"Failed to instantiate attack {attack} for model '{model_label}'.")
            continue

        if nodes is None or not isinstance(nodes, collections.Sequence) or not nodes:
            nodes = get_local_attack_nodes(attr, adj, labels, model, idx_test, device, 
                                           topk=int(nodes_topk / 4), min_node_degree=int(1 / min(epsilons)))
        nodes = [int(i) for i in nodes]

        assert all(np.unique(nodes) == np.sort(nodes)), "Attacked node list contains duplicates"
        for node in nodes:
            degree = adj[node].sum()
            for eps in epsilons:
                n_perturbations = int((eps * degree).round().item())
                if n_perturbations == 0:
                    logging.error(
                        f"Skipping attack for model '{model}' using {attack} with eps {eps} at node {node}.")
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
                    f'Evaluated model {model_label} using {attack} with pert. edges for node {node} and budget {n_perturbations}: ')

                results.append({
                    'label': model_label,
                    'epsilon': eps,
                    'n_perturbations': n_perturbations,
                    'degree': int(degree.item()),
                    'logits': logits.cpu().numpy().tolist(),
                    'initial_logits': initial_logits.cpu().numpy().tolist(),
                    'larget': labels[node].item(),
                    'node_id': node,
                    'perturbed_edges': adversary.get_perturbed_edges().cpu().numpy().tolist()
                })

                results[-1].update(adversary.classification_statistics(logits.cpu(), labels[node].long().cpu()))
                results[-1].update({
                    f'initial_{key}': value
                    for key, value
                    in adversary.classification_statistics(initial_logits.cpu(), labels[node].long().cpu()).items()
                })
                logging.info(results[-1])
                logging.info(
                    f"Completed attack and evaluation of {model_label} using {attack} with pert. edges for node {node} and budget {n_perturbations}")
                # if hasattr(adversary, 'attack_statistics'):
                #     results[-1]['attack_statistics'] = adversary.attack_statistics
        if hasattr(adversary, "ppr_matrix"):
            adversary.ppr_matrix.save_to_storage()
    assert len(results) > 0

    return {
        'results': results
    }
