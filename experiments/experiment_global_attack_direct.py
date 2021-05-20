
import logging
from typing import Any, Dict, Sequence, Union

from sacred import Experiment
import seml
import torch

from rgnn_at_scale.attacks import Attack, create_attack
from experiments.common import prepare_attack_experiment, run_global_attack

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
    make_undirected = True
    seed = 0

    attack = 'PRBCD'
    attack_params = {}
    epsilons = [0.01]

    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained_direct'
    pert_adj_storage_type = 'evasion_attack_adj'
    pert_attr_storage_type = 'evasion_attack_attr'
    model_label = "Vanilla GCN"

    device = 0
    data_device = 0
    debug_level = "info"


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float],
        binary_attr: bool, make_undirected: bool, seed: int, artifact_dir: str, pert_adj_storage_type: str,
        pert_attr_storage_type: str, model_label: str, model_storage_type: str, device: Union[str, int],
        data_device: Union[str, int], debug_level: str):

    results = []
    surrogate_model_label = False

    (
        attr, adj, labels, _, _, idx_test, storage, attack_params, pert_params, model_params, m
    ) = prepare_attack_experiment(
        data_dir, dataset, attack, attack_params, epsilons, binary_attr, make_undirected,  seed, artifact_dir,
        pert_adj_storage_type, pert_attr_storage_type, model_label, model_storage_type, device, surrogate_model_label,
        data_device, debug_level, ex
    )

    if model_label is not None and model_label:
        model_params['label'] = model_label

    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    for model, hyperparams in models_and_hyperparams:
        model_label = hyperparams["label"]
        adversary = create_attack(attack, attr=attr, adj=adj, labels=labels, model=model, idx_attack=idx_test,
                                  device=device, data_device=data_device, binary_attr=binary_attr,
                                  make_undirected=make_undirected, **attack_params)

        for epsilon in epsilons:
            run_global_attack(epsilon, m, storage, pert_adj_storage_type, pert_attr_storage_type,
                              pert_params, adversary, model_label)

            adj_adversary = adversary.adj_adversary
            attr_adversary = adversary.attr_adversary

            logits, accuracy = Attack.evaluate_global(model.to(device), attr_adversary.to(device),
                                                      adj_adversary.to(device), labels, idx_test)

            results.append({
                'label': model_label,
                'epsilon': epsilon,
                'accuracy': accuracy
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    assert len(results) > 0

    return {
        'results': results
    }
