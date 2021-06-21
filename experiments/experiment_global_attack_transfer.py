import logging
import warnings
from typing import Any, Dict, Sequence, Union

from sacred import Experiment

import torch

from rgnn_at_scale.attacks import Attack, create_attack
from experiments.common import prepare_attack_experiment, run_global_attack

try:
    import seml
except:  # noqa: E722
    seml = None

ex = Experiment()

if seml is not None:
    seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None

    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    # default params
    dataset = 'cora_ml'
    data_dir = './datasets'
    binary_attr = False
    make_undirected = True
    seed = 0

    attack = 'PGD'
    attack_params = {}
    epsilons = [0.01, 0.1, 0.5, 1.0]

    artifact_dir = 'cache_debug'
    pert_adj_storage_type = 'evasion_attack_adj'
    pert_attr_storage_type = 'evasion_attack_attr'

    model_storage_type = 'pretrained'
    model_label = None

    surrogate_model_storage_type = 'pretrained'
    surrogate_model_label = "Vanilla Dense GCN"

    device = 0
    data_device = 0
    debug_level = "info"


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float],
        binary_attr: bool, make_undirected: bool, seed: int, artifact_dir: str, pert_adj_storage_type: str,
        pert_attr_storage_type: str, model_label: str, model_storage_type: str, surrogate_model_storage_type: str,
        surrogate_model_label: str, device: Union[str, int], data_device: Union[str, int], debug_level: str):

    assert surrogate_model_label is not None, "Surrogate model label must not be None"

    results = []

    (
        attr, adj, labels, _, _, idx_test, storage, attack_params, pert_params, model_params, m
    ) = prepare_attack_experiment(
        data_dir, dataset, attack, attack_params, epsilons, binary_attr, make_undirected, seed, artifact_dir,
        pert_adj_storage_type, pert_attr_storage_type, model_label, model_storage_type, device, surrogate_model_label,
        data_device, debug_level, ex
    )

    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    model_params["label"] = surrogate_model_label
    surrogate_models_and_hyperparams = storage.find_models(surrogate_model_storage_type, model_params)

    assert len(models_and_hyperparams) > 0, "No evaluation models found!"
    assert len(surrogate_models_and_hyperparams) > 0, "No surrogate model found!"
    if len(surrogate_models_and_hyperparams) > 1:
        warnings.warn("More than one matching surrogate model found. Choose last one by default.")

    surrogate_model = surrogate_models_and_hyperparams[-1][0]

    for epsilon in epsilons:
        adversary = create_attack(attack, attr=attr, adj=adj, labels=labels, model=surrogate_model,
                                  idx_attack=idx_test, device=device, data_device=data_device,
                                  binary_attr=binary_attr, make_undirected=make_undirected, **attack_params)

        run_global_attack(epsilon, m, storage, pert_adj_storage_type, pert_attr_storage_type,
                          pert_params, adversary, surrogate_model_label)

        # Clear to save GPU memory
        adj_adversary = adversary.adj_adversary
        attr_adversary = adversary.attr_adversary
        del adversary

        for model, hyperparams in models_and_hyperparams:
            current_label = hyperparams["label"]
            logging.info(f"Evaluate  {attack} for model '{current_label}'.")
            logits, accuracy = Attack.evaluate_global(model.to(device), attr_adversary.to(device),
                                                      adj_adversary.to(device), labels, idx_test)

            results.append({
                'label': current_label,
                'epsilon': epsilon,
                'accuracy': accuracy
            })

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    assert len(results) > 0

    return {
        'results': results
    }
