from typing import Any, Dict, Sequence, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sacred import Experiment
from torch_sparse import SparseTensor

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.models import BATCHED_PPR_MODELS
from rgnn_at_scale.helper.utils import accuracy


def prepare_attack_experiment(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any],
                              epsilons: Sequence[float], binary_attr: bool, make_undirected: bool,
                              seed: int, artifact_dir: str, pert_adj_storage_type: str, pert_attr_storage_type: str,
                              model_label: str, model_storage_type: str, device: Union[str, int],
                              surrogate_model_label: str, data_device: Union[str, int], debug_level: str,
                              ex: Experiment):

    if debug_level is not None and isinstance(debug_level, str):
        logger = logging.getLogger()
        if debug_level.lower() == "info":
            logger.setLevel(logging.INFO)
        if debug_level.lower() == "debug":
            logger.setLevel(logging.DEBUG)
        if debug_level.lower() == "critical":
            logger.setLevel(logging.CRITICAL)
        if debug_level.lower() == "error":
            logger.setLevel(logging.ERROR)

    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not availble, set device to 'cpu'"
        assert data_device == "cpu", "CUDA is not availble, set device to 'cpu'"

    logging.info({
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'epsilons': epsilons,
        'make_undirected': make_undirected, 'binary_attr': binary_attr, 'seed': seed,
        'artifact_dir':  artifact_dir, 'pert_adj_storage_type': pert_adj_storage_type,
        'pert_attr_storage_type': pert_attr_storage_type, 'model_label': model_label,
        'model_storage_type': model_storage_type, 'device': device, 'data_device': data_device
    })

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'

    # To increase consistency between runs
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                       binary_attr=binary_attr, return_original_split=dataset.startswith('ogbn'))

    attr, adj, labels = graph[:3]
    if graph[3] is None:
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    storage = Storage(artifact_dir, experiment=ex)

    attack_params = dict(attack_params)
    if "ppr_cache_params" in attack_params.keys():
        ppr_cache_params = dict(attack_params["ppr_cache_params"])
        ppr_cache_params['dataset'] = dataset
        attack_params["ppr_cache_params"] = ppr_cache_params

    pert_params = dict(dataset=dataset,
                       binary_attr=binary_attr,
                       make_undirected=make_undirected,
                       seed=seed,
                       attack=attack,
                       model=model_label if model_label == surrogate_model_label else None,  # For legacy reasons
                       surrogate_model=surrogate_model_label,
                       attack_params=attack_params)

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr,
                        make_undirected=make_undirected,
                        seed=seed)

    if model_label is not None and model_label:
        model_params["label"] = model_label

    if make_undirected:
        m = adj.nnz() / 2
    else:
        m = adj.nnz()

    return attr, adj, labels, idx_train, idx_val, idx_test, storage, attack_params, pert_params, model_params, m


def run_global_attack(epsilon, m, storage, pert_adj_storage_type, pert_attr_storage_type,
                      pert_params, adversary, model_label):
    n_perturbations = int(round(epsilon * m))

    pert_adj = storage.load_artifact(pert_adj_storage_type, {**pert_params, **{'epsilon': epsilon}})
    pert_attr = storage.load_artifact(pert_attr_storage_type, {**pert_params, **{'epsilon': epsilon}})

    if pert_adj is not None and pert_attr is not None:
        logging.info(
            f"Found cached perturbed adjacency and attribute matrix for model '{model_label}' and eps {epsilon}")
        adversary.set_pertubations(pert_adj, pert_attr)
    else:
        logging.info(f"No cached perturbations found for model '{model_label}' and eps {epsilon}. Execute attack...")
        adversary.attack(n_perturbations)
        pert_adj, pert_attr = adversary.get_pertubations()

        if n_perturbations > 0:
            storage.save_artifact(pert_adj_storage_type, {**pert_params, **{'epsilon': epsilon}}, pert_adj)
            storage.save_artifact(pert_attr_storage_type, {**pert_params, **{'epsilon': epsilon}}, pert_attr)


def sample_attack_nodes(logits: torch.Tensor, labels: torch.Tensor, nodes_idx,
                        adj: SparseTensor, topk: int, min_node_degree: int):
    assert logits.shape[0] == labels.shape[0]
    if isinstance(nodes_idx, torch.Tensor):
        nodes_idx = nodes_idx.cpu()
    node_degrees = adj[nodes_idx.tolist()].sum(-1)

    suitable_nodes_mask = (node_degrees >= min_node_degree).cpu()

    labels = labels.cpu()[suitable_nodes_mask]
    confidences = F.softmax(logits.cpu()[suitable_nodes_mask], dim=-1)

    correctly_classifed = confidences.max(-1).indices == labels

    logging.info(
        f"Found {sum(suitable_nodes_mask)} suitable '{min_node_degree}+ degree' nodes out of {len(nodes_idx)} "
        f"candidate nodes to be sampled from for the attack of which {correctly_classifed.sum().item()} have the "
        "correct class label")

    assert sum(suitable_nodes_mask) >= (topk * 4), \
        f"There are not enough suitable nodes to sample {(topk*4)} nodes from"

    _, max_confidence_nodes_idx = torch.topk(confidences[correctly_classifed].max(-1).values, k=topk)
    _, min_confidence_nodes_idx = torch.topk(-confidences[correctly_classifed].max(-1).values, k=topk)

    rand_nodes_idx = np.arange(correctly_classifed.sum().item())
    rand_nodes_idx = np.setdiff1d(rand_nodes_idx, max_confidence_nodes_idx)
    rand_nodes_idx = np.setdiff1d(rand_nodes_idx, min_confidence_nodes_idx)
    rnd_sample_size = min((topk * 2), len(rand_nodes_idx))
    rand_nodes_idx = np.random.choice(rand_nodes_idx, size=rnd_sample_size, replace=False)

    return (np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][max_confidence_nodes_idx])[None].flatten(),
            np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][min_confidence_nodes_idx])[None].flatten(),
            np.array(nodes_idx[suitable_nodes_mask][correctly_classifed][rand_nodes_idx])[None].flatten())


def get_local_attack_nodes(attr, adj, labels, surrogate_model, idx_test, device, topk=10, min_node_degree=2):

    with torch.no_grad():
        surrogate_model = surrogate_model.to(device)
        surrogate_model.eval()
        if type(surrogate_model) in BATCHED_PPR_MODELS.__args__:
            logits = surrogate_model.forward(attr, adj, ppr_idx=np.array(idx_test))
        else:
            logits = surrogate_model(attr.to(device), adj.to(device))[idx_test]

        acc = accuracy(logits.cpu(), labels.cpu()[idx_test], np.arange(logits.shape[0]))

    logging.info(f"Sample Attack Nodes for model with accuracy {acc:.4}")

    max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx = sample_attack_nodes(
        logits, labels[idx_test], idx_test, adj, topk,  min_node_degree)
    tmp_nodes = np.concatenate((max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx))
    logging.info(
        f"Sample the following attack nodes:\n{max_confidence_nodes_idx}\n{min_confidence_nodes_idx}\n{rand_nodes_idx}")
    return tmp_nodes
