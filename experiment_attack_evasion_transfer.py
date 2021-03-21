import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
from sacred import Experiment
import seml
import torch

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.attacks import create_attack, SPARSE_ATTACKS
from rgnn_at_scale.io import Storage
from rgnn_at_scale.models import DenseGCN, GCN
from rgnn_at_scale.train import train
from rgnn_at_scale.utils import accuracy


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
    attack = 'FGSM'
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
    pert_adj_storage_type = 'evasion_transfer_attack_adj'
    pert_attr_storage_type = 'evasion_transfer_attack_attr'
    model_storage_type = 'pretrained'
    device = 0
    display_steps = 10
    model_label = None
    make_undirected = True
    make_unweighted = True


@ex.automain
def run(data_dir: str, dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float], binary_attr: bool,
        make_undirected: bool, make_unweighted: bool,  normalize: bool, normalize_attr: str, surrogate_params: Dict[str, Any], seed: int, 
        artifact_dir: str, pert_adj_storage_type: str, pert_attr_storage_type: str, model_label: str, model_storage_type: str,
        device: Union[str, int], data_device: Union[str, int], display_steps: int):
    logging.info({
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'epsilons': epsilons,
        'normalize': normalize, 'normalize_attr': normalize_attr,
        'binary_attr': binary_attr, 'surrogate_params': surrogate_params, 'seed': seed,
        'artifact_dir': artifact_dir, 'pert_adj_storage_type': pert_adj_storage_type,
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
                  make_undirected=make_undirected,
                  make_unweighted=make_unweighted,
                  seed=seed,
                  attack=attack,
                  surrogate_params=surrogate_params,
                  attack_params=attack_params)

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

    if len(adj_per_eps) == 0:
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

        adversary = create_attack(attack, binary_attr, attr, adj=adj, labels=labels,
                                  model=gcn, idx_attack=idx_test, device=device, **attack_params)

        tmp_epsilons = list(epsilons)
        if tmp_epsilons[0] != 0:
            tmp_epsilons.insert(0, 0)

        m = adj.nnz() / 2
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

        del adversary

    adj = adj.to('cpu')
    attr = attr.to('cpu')
    if epsilons[0] == 0:
        adj_per_eps.insert(0, adj)
        attr_per_eps.insert(0, attr)

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr,
                        seed=seed)

    if model_label is not None and model_label:
        model_params['label'] = model_label
    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    results = []
    with torch.no_grad():
        for model, hyperparams in models_and_hyperparams:
            model = model.to(device)
            model.eval()

            if hasattr(model, 'release_cache'):
                model.release_cache()

            for eps, adj_perturbed, attr_perturbed in zip(epsilons, adj_per_eps, attr_per_eps):
                # In case the model is non-deterministic to get the results either after attacking or after loading
                torch.manual_seed(seed)
                np.random.seed(seed)

<<<<<<< HEAD
                pred_logits_target = model(attr_perturbed.to(device), adj_perturbed.to(device))
                acc_test_target = accuracy(pred_logits_target.cpu(), labels.cpu(), idx_test)
                results.append({
                    'label': hyperparams['label'],
                    'epsilon': eps,
                    'accuracy': acc_test_target
                })
=======
                try:

                    pred_logits_target = model(attr_perturbed.to(device), adj_perturbed.to(device))
                    acc_test_target = accuracy(pred_logits_target, labels.to(device), idx_test)
                    results.append({
                        'label': hyperparams['label'],
                        'epsilon': eps,
                        'accuracy': acc_test_target
                    })

                except Exception as e:
                    logging.error(f'Failed for {hyperparams["label"]} and epsilon {eps}')
                    logging.exception(e)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
>>>>>>> main

    return {
        'results': results
    }
