import collections
from copy import deepcopy
import logging
import warnings
from typing import Any, Dict, Sequence, Union

import numpy as np
from sacred import Experiment
import torch

from experiments.common import get_local_attack_nodes, prepare_attack_experiment
from rgnn_at_scale.attacks import create_attack
from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.train import train


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
    attack = 'Nettack'
    attack_params = {}
    nodes = None
    nodes_topk = 40

    epsilons = [1]
    min_node_degree = None
    seed = 0

    artifact_dir = "cache"

    model_storage_type = 'pretrained'
    model_label = "Vanilla GCN"

    surrogate_model_storage_type = "pretrained_linear"
    surrogate_model_label = 'Linear GCN'

    data_dir = './data'
    binary_attr = False
    make_undirected = True

    data_device = 'cpu'
    device = 0
    debug_level = "info"

    evaluate_poisoning = True


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], nodes: str, seed: int,
        epsilons: Sequence[float], min_node_degree: int, binary_attr: bool, make_undirected: bool, artifact_dir: str, nodes_topk: int,
        model_label: str, model_storage_type: str, device: Union[str, int], surrogate_model_storage_type: str,
        surrogate_model_label: str, data_device: Union[str, int], debug_level: str, evaluate_poisoning: bool):
    """
    Instantiates a sacred experiment executing a local transfer attack run for a given model configuration.
    Local evasion attacks aim to flip the label of a single node only.
    Transfer attacks are used to attack a model via a surrogate model.

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
            - LocalPRBCD
            - LocalDICE
            - Nettack
            - SGA
    attack_params : Dict[str, Any], optional
        The attack hyperparams to be passed as keyword arguments to the constructor of the attack class
    epsilons: List[float]
        The budgets for which the attack on the model should be executed.
    nodes: List[int], optional
        The IDs of the nodes to be attacked.
    nodes_topk: int
        The number of nodes to be sampled if the nodes parameter is None.
        Nodes are sampled to include 25% high-confidence, 25% low-confidence and 50% random nodes.
        When sampling nodes, only nodes with a degree >= 1/(min(epsilons)) are considered. 
    min_node_degree: int, optional
        When sampling nodes this overwrite the degree >= 1/(min(epsilons)) constraint to only sample
        nodes with degree >= min_node_degree. Use this to make sure multiple independent runs of this
        experiment with different epsilons are comparable. 
    model_label : str, optional
        The name given to the model (to be attack) using the experiment_train.py 
        This name is used as an identifier in combination with the dataset configuration to retrieve 
        the model to be attacked from storage. If None, all models that were fit on the given dataset 
        are attacked.
    surrogate_model_label : str, optional
        Same as model_label but for the model used as surrogate for the attack.
    artifact_dir: str
        The path to the folder that acts as TinyDB Storage for pretrained models
    model_storage_type: str
        The name of the storage (TinyDB) table name the model to be attacked is retrieved from.
    surrogate_model_storage_type: str
        The name of the storage (TinyDB) table name the surrogate model is retrieved from.
    pert_adj_storage_type: str
        The name of the storage (TinyDB) table name the perturbed adjacency matrix is stored to
    pert_attr_storage_type: str
        The name of the storage (TinyDB) table name the perturbed attribute matrix is stored to
    evaluate_poisoning: bool
        If set to `True` also the poisoning performance will be evaluated

    Returns
    -------
    List[Dict[str, any]]
        List of result dictionaries. One for every combination of model and epsilon.
        Each result dictionary contains the model labels, epsilon value and the perturbed accuracy
    """

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'
    results = []

    (
        attr, adj, labels, idx_train, idx_val, idx_test, storage, attack_params, _, model_params, _
    ) = prepare_attack_experiment(
        data_dir, dataset, attack, attack_params, epsilons, binary_attr, make_undirected, seed, artifact_dir,
        None, None, model_label, model_storage_type, device, surrogate_model_label, data_device, debug_level, ex
    )

    storage = Storage(artifact_dir, experiment=ex)

    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    model_params["label"] = surrogate_model_label
    surrogate_models_and_hyperparams = storage.find_models(surrogate_model_storage_type, model_params)

    assert len(surrogate_models_and_hyperparams) > 0, "No surrogate model found!"
    if len(surrogate_models_and_hyperparams) > 1:
        warnings.warn("More than one matching surrogate model found. Choose last one by default.")

    surrogate_model = surrogate_models_and_hyperparams[-1][0]

    logging.error(f"Found {len(models_and_hyperparams)} models with label '{model_label}' to attack.")
    for model, hyperparams in models_and_hyperparams:
        eval_model_label = hyperparams['label']

        adversary = create_attack(attack, attr=attr, adj=adj, labels=labels, model=surrogate_model,
                                  idx_attack=idx_test, device=device,  data_device=data_device,
                                  binary_attr=binary_attr, make_undirected=make_undirected, **attack_params)
        adversary.set_eval_model(model)
        if hasattr(adversary, "ppr_matrix"):
            adversary.ppr_matrix.save_to_storage()

        tmp_nodes = np.array(nodes)
        if nodes is None or not isinstance(nodes, collections.Sequence) or not nodes:
            minimal_degree = min_node_degree
            if minimal_degree is None:
                minimal_degree = int(1 / min(epsilons))

            assert minimal_degree >= int(
                1 / min(epsilons)), f"The min_node_degree has to be smaller than 'int(1 / min(epsilons)' == {int(1 / min(epsilons))}"

            tmp_nodes = get_local_attack_nodes(attr, adj, labels, model,
                                               idx_test, device,  topk=int(nodes_topk / 4), min_node_degree=minimal_degree)
        tmp_nodes = [int(i) for i in tmp_nodes]
        for node in tmp_nodes:
            degree = adj[node].sum()
            for eps in epsilons:
                n_perturbations = int((eps * degree).round().item())
                if n_perturbations == 0:
                    logging.error(
                        f"Skipping attack for model '{surrogate_model}' using {attack} with eps {eps} at node {node}.")
                    continue

                print(f'MAX MEMORY BEFORE: {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1e9}')

                adversary.attack(n_perturbations, node_idx=node)

                print(f'MAX MEMORY AFTER: {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1e9}')

                logits_evasion, initial_logits_evasion = adversary.evaluate_local(node)

                logging.info(
                    f'Evaluated model {eval_model_label} using {attack} with pert. edges for node {node} and budget {n_perturbations}:')

                results.append({
                    'label': model_label,
                    'epsilon': eps,
                    'n_perturbations': n_perturbations,
                    'degree': int(degree.item()),
                    'target': labels[node].item(),
                    'node_id': node,
                    'perturbed_edges': adversary.get_perturbed_edges().cpu().numpy().tolist(),
                    'evasion': {
                        'logits': logits_evasion.cpu().numpy().tolist(),
                        'initial_logits': initial_logits_evasion.cpu().numpy().tolist(),
                        **adversary.classification_statistics(
                            logits_evasion.cpu(), labels[node].long().cpu()),
                        **{
                            f'initial_{key}': value
                            for key, value
                            in adversary.classification_statistics(
                                initial_logits_evasion.cpu(), labels[node].long().cpu()).items()
                        }
                    }
                })

                if evaluate_poisoning:
                    victim = deepcopy(model).to(device)
                    for module in victim.modules():
                        if hasattr(module, 'reset_parameters'):
                            module.reset_parameters()

                    adj_adversary = adversary.adj_adversary_for_poisoning()

                    if hasattr(victim, 'fit'):
                        if hasattr(victim, 'ppr_cache_params'):
                            victim.ppr_cache_params = None

                        _ = victim.fit(adj_adversary.to(device), attr.to(device),
                                       labels=labels.to(device),
                                       idx_train=idx_train,
                                       idx_val=idx_val,
                                       dataset=dataset,
                                       make_undirected=make_undirected,
                                       **hyperparams['train_params'])
                    else:
                        _ = train(
                            model=victim, attr=attr.to(device), adj=adj_adversary.to(device),
                            labels=labels.to(device), idx_train=idx_train, idx_val=idx_val, **hyperparams['train_params']
                        )

                    adversary.set_eval_model(victim)
                    logits_poisoning, _ = adversary.evaluate_local(node)
                    adversary.set_eval_model(model)

                    results[-1]['poisoning'] = {
                        'logits': logits_poisoning.cpu().numpy().tolist(),
                        'initial_logits': initial_logits_evasion.cpu().numpy().tolist(),
                        **adversary.classification_statistics(
                            logits_poisoning.cpu(), labels[node].long().cpu()),
                        **{
                            f'initial_{key}': value
                            for key, value
                            in adversary.classification_statistics(
                                initial_logits_evasion.cpu(), labels[node].long().cpu()).items()
                        }
                    }

                logging.info(results[-1])

        if hasattr(adversary, "ppr_matrix"):
            adversary.ppr_matrix.save_to_storage()
    assert len(results) > 0

    return {
        'results': results
    }
