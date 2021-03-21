import logging
from typing import Any, Dict, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from sacred import Experiment
from scipy.sparse import csgraph
import seml
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, remove_isolated_nodes, subgraph, to_networkx

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.attacks import create_attack, SPARSE_ATTACKS
from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.models import DenseGCN, GCN
from rgnn_at_scale.train import train
from rgnn_at_scale.helper.utils import accuracy


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
    dataset = 'ogbn-arxiv'  # Options are 'cora_ml' and 'citeseer' (or with a big GPU 'pubmed')
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
    seed = 0
    artifact_dir = 'cache_debug'
    pert_adj_storage_type = 'evasion_transfer_attack_vs_number_nodes_adj'
    pert_attr_storage_type = 'evasion_transfer_attack_vs_number_nodes_attr'
    device = 0
    display_steps = 10
    model_label = None

    n_classes = 8
    use_largest_component = False


def _isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


@ex.automain
def run(dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float], binary_attr: bool,
        surrogate_params: Dict[str, Any], seed: int, artifact_dir: str, pert_adj_storage_type: str,
        pert_attr_storage_type: str, device: Union[str, int], display_steps: int,
        model_label: str, n_classes: int, use_largest_component: bool):
    logging.info({
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'epsilons': epsilons,
        'binary_attr': binary_attr, 'surrogate_params': surrogate_params, 'seed': seed,
        'artifact_dir': artifact_dir, 'pert_adj_storage_type': pert_adj_storage_type,
        'pert_attr_storage_type': pert_attr_storage_type, 'model_label': model_label, 'n_classes': n_classes,
        'device': device, 'display_steps': display_steps, 'use_largest_component': use_largest_component
    })

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'
    assert 'train_params' in surrogate_params, '`surrogate` must contain the field `train_params`'

    results = []

    graph = prep_graph(dataset, device, binary_attr=binary_attr, return_original_split=dataset.startswith('ogbn'))
    attr_complete, adj_complete, labels_complete = graph[:3]
    if len(graph) == 3:
        idx_train, idx_val, idx_test = split(labels_complete.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']
    n_features = attr_complete.shape[1]

    idx_train_complete = torch.zeros(attr_complete.size(0), dtype=torch.bool)
    idx_val_complete = torch.zeros(attr_complete.size(0), dtype=torch.bool)
    idx_test_complete = torch.zeros(attr_complete.size(0), dtype=torch.bool)
    idx_train_complete[idx_train], idx_val_complete[idx_val], idx_test_complete[idx_test] = (1, 1, 1)

    label_id = torch.unique(labels_complete)
    perm = torch.randperm(label_id.size(0))
    selected_labels, indices = label_id[perm[:n_classes]].sort()

    mask_current_nodes = _isin(labels_complete, selected_labels)
    n_nodes = mask_current_nodes.sum().item()

    label_map = torch.zeros_like(label_id)
    label_map[selected_labels] = indices
    labels = label_map[labels_complete[mask_current_nodes]]
    _, label_count = labels.unique(return_counts=True)

    adj = torch.sparse.FloatTensor(
        *subgraph(mask_current_nodes, adj_complete._indices(), adj_complete._values(), relabel_nodes=True),
        (n_nodes, n_nodes)
    ).coalesce()
    attr = attr_complete[mask_current_nodes, :]

    edge_index_lcc, edge_values_lcc, mask_lcc = remove_isolated_nodes(adj._indices(), adj._values())
    label_id_lcc, label_count_lcc = labels[mask_lcc].unique(return_counts=True)

    graph_statistics = {
        'n_nodes': adj.shape[0],
        'n_nodes_largest_component': mask_lcc.sum().item(),
        'average_degree': degree(adj._indices()[0]).mean().item(),
        'average_degree_largest_component': degree(edge_index_lcc[0]).mean().item(),
        'n_classes': label_id.shape[0],
        'n_classes_largest_component': label_id_lcc.shape[0],
        'label_count': label_count.cpu().tolist(),
        'label_count_largest_component': label_count_lcc.cpu().tolist(),
        'sparsity': adj._nnz() / (adj.shape[0] * adj.shape[1])
    }

    idx_train, idx_val, idx_test = idx_train_complete[mask_current_nodes], idx_val_complete[mask_current_nodes], idx_test_complete[mask_current_nodes]

    if use_largest_component:
        n_nodes = mask_lcc.sum()
        adj = torch.sparse.FloatTensor(
            *subgraph(mask_lcc, adj._indices(), adj._values(), relabel_nodes=True),
            (n_nodes, n_nodes)
        ).coalesce()
        attr = attr[mask_lcc, :]
        idx_train, idx_val, idx_test = idx_train[mask_lcc], idx_val[mask_lcc], idx_test[mask_lcc]
        labels = labels[mask_lcc]

    idx_train, idx_val, idx_test = idx_train.nonzero().squeeze(), idx_val.nonzero().squeeze(), idx_test.nonzero().squeeze()

    graph_statistics['label_count_train'] = labels[idx_train].unique(return_counts=True)[1].cpu().tolist()

    params = dict(dataset=dataset, binary_attr=binary_attr, seed=seed, attack=attack,
                  surrogate_params=surrogate_params, attack_params=attack_params,
                  use_largest_component=use_largest_component, n_classes=n_classes)
    storage = Storage(artifact_dir, experiment=ex)

    adj_per_eps = []
    attr_per_eps = []
    for epsilon in epsilons:
        if epsilon == 0:
            continue

        pert_adj = storage.load_artifact(pert_adj_storage_type, {**params, **{'epsilon': epsilon}})
        pert_attr = storage.load_artifact(pert_attr_storage_type, {**params, **{'epsilon': epsilon}})
        if pert_adj is None or pert_attr is None:
            # Due to the greedy fashion we only use the existing adjacency matrices if all do exist
            adj_per_eps = []
            attr_per_eps = []
            break

        adj_per_eps.append(pert_adj)
        attr_per_eps.append(pert_attr)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if attack in SPARSE_ATTACKS:
        gcn = GCN(n_classes=n_classes, n_features=n_features, **surrogate_params).to(device)
        adj_surrogate = adj
    else:
        gcn = DenseGCN(n_classes=n_classes, n_features=n_features, **surrogate_params).to(device)
        adj_surrogate = adj.to_dense()
    train(model=gcn, attr=attr.to(device), adj=adj_surrogate.to(device), labels=labels.to(device),
          idx_train=idx_train, idx_val=idx_val, display_step=display_steps, **surrogate_params['train_params'])
    gcn.eval()
    if hasattr(gcn, 'release_cache'):
        gcn.release_cache()
    with torch.no_grad():
        pred_logits_surr = gcn(attr.to(device), adj_surrogate.to(device))
    logging.info(f'Test accuracy of surrogate: {accuracy(pred_logits_surr, labels.to(device), idx_test)}')
    del pred_logits_surr

    if len(adj_per_eps) == 0:
        adversary = create_attack(attack, binary_attr, attr, adj=adj, labels=labels,
                                  model=gcn, idx_attack=idx_test, device=device, **attack_params)

        tmp_epsilons = list(epsilons)
        if tmp_epsilons[0] != 0:
            tmp_epsilons.insert(0, 0)

        m = adj._nnz() / 2
        for epsilon in tmp_epsilons[1:]:
            logging.info(f'Attack via {attack} with budget {epsilon}')

            # To increase consistency between runs
            torch.manual_seed(seed)
            np.random.seed(seed)

            n_perturbations = int(round(epsilon * m))
            adversary.attack(n_perturbations)
            adj_per_eps.append(adversary.adj_adversary.cpu())
            attr_per_eps.append(adversary.attr_adversary.cpu())

            storage.save_artifact(pert_adj_storage_type, {**params, **{'epsilon': epsilon}}, adj_per_eps[-1])
            storage.save_artifact(pert_attr_storage_type, {**params, **{'epsilon': epsilon}}, attr_per_eps[-1])

    if epsilons[0] == 0:
        adj_per_eps.insert(0, adj.to('cpu'))
        attr_per_eps.insert(0, attr.to('cpu'))

    with torch.no_grad():
        model = gcn.to(device)
        model.eval()

        for eps, adj_perturbed, attr_perturbed in zip(epsilons, adj_per_eps, attr_per_eps):
            # In case the model is non-deterministic to get the results either after attacking or after loading
            torch.manual_seed(seed)
            np.random.seed(seed)

            pred_logits_target = model(attr_perturbed.to(device), adj_perturbed.to(device))
            acc_test_target = accuracy(pred_logits_target, labels.to(device), idx_test)
            results.append({
                'label': 'Surrogate',
                'epsilon': eps,
                'accuracy': acc_test_target,
                'labels': selected_labels.cpu().tolist(),
                **graph_statistics
            })

    return {
        'results': results
    }
