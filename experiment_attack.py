import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
from sacred import Experiment
import seml
import torch

from rgnn.data import prep_graph, split
from rgnn.fgsm import FGSM
from rgnn.io import Storage
from rgnn.models import DenseGCN
from rgnn.pgd import PGD
from rgnn.train import train
from rgnn.utils import accuracy


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
    attack = 'fgsm'  # Options are 'fgsm' and 'pgd'
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
    artifact_dir = 'cache'  # 'cache_debug'
    pert_adj_storage_type = 'attack'
    model_storage_type = 'pretrained'
    device = 0
    display_steps = 10


@ex.automain
def run(dataset: str, attack: str, attack_params: Dict[str, Any], epsilons: Sequence[float], binary_attr: bool,
        surrogate_params: Dict[str, Any], seed: int, artifact_dir: str, pert_adj_storage_type: str,
        model_storage_type: str, device: Union[str, int], display_steps: int):
    logging.info({
        'dataset': dataset, 'attack': attack, 'attack_params': attack_params, 'epsilons': epsilons,
        'binary_attr': binary_attr, 'surrogate_params': surrogate_params, 'seed': seed,
        'artifact_dir': artifact_dir, 'pert_adj_storage_type': pert_adj_storage_type,
        'model_storage_type': model_storage_type, 'device': device, 'display_steps': display_steps
    })

    binary_attr = False

    assert sorted(epsilons) == epsilons, 'argument `epsilons` must be a sorted list'
    assert len(np.unique(epsilons)) == len(epsilons),\
        'argument `epsilons` must be unique (strictly increasing)'
    assert all([eps >= 0 for eps in epsilons]), 'all elements in `epsilons` must be greater than 0'
    assert 'train_params' in surrogate_params, '`surrogate` must contain the field `train_params`'

    results = []

    attr, adj, labels = prep_graph(dataset, device=device, binary_attr=binary_attr)
    n_features = attr.shape[1]
    n_classes = int(labels.max() + 1)

    idx_train, idx_val, idx_test = split(labels.cpu().numpy())

    params = dict(dataset=dataset, binary_attr=binary_attr, seed=seed, attack=attack, attack_params=attack_params)
    storage = Storage(artifact_dir)

    adj_per_eps = []
    for epsilon in epsilons:
        if epsilon == 0:
            continue
        pert_adj = storage.load_sparse_tensor(pert_adj_storage_type, {**params, **{'epsilon': epsilon}})
        if pert_adj is None:
            # Due to the greedy fashion we only use the existing adjacency matrices if all do exist
            adj_per_eps = []
            break
        else:
            adj_per_eps.append(pert_adj)

    if len(adj_per_eps) == 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        gcn = DenseGCN(n_classes=n_classes, n_features=n_features, **surrogate_params).to(device)
        _ = train(model=gcn, attr=attr, adj=adj.to_dense(), labels=labels, idx_train=idx_train, idx_val=idx_val,
                  display_step=display_steps, **surrogate_params['train_params'])
        gcn.eval()
        with torch.no_grad():
            pred_logits_surr = gcn(attr, adj.to_dense())
        logging.info(f'Test accuracy of surrogate: {accuracy(pred_logits_surr, labels, idx_test)}')

        if attack == 'pgd':
            adversary = PGD(adj=adj, X=attr, labels=labels, model=gcn, idx_attack=idx_test, **attack_params)
        else:
            adversary = FGSM(adj=adj, X=attr, labels=labels, model=gcn, idx_attack=idx_test, **attack_params)

        tmp_epsilons = list(epsilons)
        if tmp_epsilons[0] != 0:
            tmp_epsilons.insert(0, 0)

        m = adj._nnz() / 2
        for eps1, eps2 in zip(tmp_epsilons[:-1], tmp_epsilons[1:]):
            logging.info(f'Attack via {attack} with budget {eps2}')

            # To increase consistency between runs
            torch.manual_seed(seed)
            np.random.seed(seed)

            n_perturbations = int(round(eps2 * m)) - int(round(eps1 * m))
            adversary.attack(n_perturbations)
            adj_per_eps.append(adversary.adj_adversary.cpu())

            storage.save_sparse_tensor(pert_adj_storage_type, {**params, **{'epsilon': eps2}}, adj_per_eps[-1])

    if epsilons[0] == 0:
        adj_per_eps.insert(0, adj.to('cpu'))

    models_and_hyperparams = storage.find_models(model_storage_type,
                                                 dict(dataset=dataset, binary_attr=binary_attr, seed=seed))

    with torch.no_grad():
        for model, hyperparams in models_and_hyperparams:
            model = model.to(device)
            model.eval()

            for eps, adj_perturbed in zip(epsilons, adj_per_eps):
                # In case the model is non-deterministic to get the results either after attacking or after loading
                torch.manual_seed(seed)
                np.random.seed(seed)

                pred_logits_target = model(attr, adj_perturbed.to(device))
                acc_test_target = accuracy(pred_logits_target, labels, idx_test)
                results.append({
                    'label': hyperparams['label'],
                    'epsilon': eps,
                    'accuracy': acc_test_target
                })

    return {
        'results': results
    }
