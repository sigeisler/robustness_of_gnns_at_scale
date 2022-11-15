
import logging
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
    data_dir = './data'
    dataset = 'cora_ml'
    make_undirected = True
    binary_attr = False
    data_device = 0

    device = 0
    seed = 0

    attack = 'PRBCD'
    attack_params = dict(
        epochs=500,
        fine_tune_epochs=100,
        keep_heuristic="WeightOnly",
        block_size=100_000,
        do_synchronize=True,
        loss_type="tanhMargin",
    )
    epsilons = [0.01, 0.1]

    artifact_dir = 'cache'
    model_label = "Soft Median GDC (T=0.5)"
    model_storage_type = 'pretrained'
    pert_adj_storage_type = 'evasion_global_adj'
    pert_attr_storage_type = 'evasion_global_attr'

    debug_level = "info"


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float],
        binary_attr: bool, make_undirected: bool, seed: int, artifact_dir: str, pert_adj_storage_type: str,
        pert_attr_storage_type: str, model_label: str, model_storage_type: str, device: Union[str, int],
        data_device: Union[str, int], debug_level: str):
    """
    Instantiates a sacred experiment executing a global direct attack run for a given model configuration.
    Caches the perturbed adjacency to storage and evaluates the models perturbed accuracy. 
    Global evasion attacks allow all nodes of the graph to be perturbed under the given budget.
    Direct attacks are used to attack a model without the use of a surrogate model.

    Parameters
    ----------
    data_dir : str
        Path to data folder that contains the dataset
    dataset : str
        Name of the dataset. Either one of: `cora_ml`, `citeseer`, `pubmed` or an ogbn dataset
    device : Union[int, torch.device]
        The device to use for training. Must be `cpu` or GPU id
    data_device : Union[int, torch.device]
        The device to use for storing the dataset. For batched models (like PPRGo) this may differ from the device parameter. 
        In all other cases device takes precedence over data_device
    make_undirected : bool
        Normalizes adjacency matrix with symmetric degree normalization (non-scalable implementation!)
    binary_attr : bool
        If true the attributes are binarized (!=0)
    attack : str
        The name of the attack class to use. Supported attacks are:
            - PRBCD
            - GreedyRBCD
            - DICE
            - FGSM
            - PGD
    attack_params : Dict[str, Any], optional
        The attack hyperparams to be passed as keyword arguments to the constructor of the attack class
    epsilons: List[float]
        The budgets for which the attack on the model should be executed.
    model_label : str, optional
        The name given to the model at train time using the experiment_train.py 
        This name is used as an identifier in combination with the dataset configuration to retrieve 
        the model to be attacked from storage. If None, all models that were fit on the given dataset 
        are attacked.
    artifact_dir: str
        The path to the folder that acts as TinyDB Storage for pretrained models
    model_storage_type: str
        The name of the storage (TinyDB) table name the model to be attacked is retrieved from.
    pert_adj_storage_type: str
        The name of the storage (TinyDB) table name the perturbed adjacency matrix is stored to
    pert_attr_storage_type: str
        The name of the storage (TinyDB) table name the perturbed attribute matrix is stored to

    Returns
    -------
    List[Dict[str, any]]
        List of result dictionaries. One for every combination of model and epsilon.
        Each result dictionary contains the model labels, epsilon value and the perturbed accuracy
    """

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
        logging.info(f"Evaluate  {attack} for model '{model_label}'.")
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
