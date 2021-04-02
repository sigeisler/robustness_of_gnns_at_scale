import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch

from rgnn_at_scale.attacks import create_attack, SPARSE_ATTACKS
from rgnn_at_scale.models import DenseGCN, GCN
from rgnn_at_scale.train import train
from rgnn_at_scale.helper.utils import accuracy


def load_perturbed_data_if_exists(storage, pert_adj_storage_type, pert_attr_storage_type, params, epsilons):
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
    return adj_per_eps, attr_per_eps


def train_surrogate_model(attack, adj, attr, labels, idx_train, idx_val, idx_test, n_classes, n_features, surrogate_params, display_steps, seed, device):
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

    if hasattr(gcn, 'release_cache'):
        gcn.release_cache()

    with torch.no_grad():
        gcn.eval()
        pred_logits_surr = gcn(attr.to(device), adj_surrogate.to(device))

    logging.info(f'Test accuracy of surrogate: {accuracy(pred_logits_surr, labels.to(device), idx_test)}')
    del pred_logits_surr

    return gcn


def run_attacks(attack, epsilons, binary_attr, attr, adj, labels, model, idx_attack, attack_params,
                params, storage, pert_adj_storage_type, pert_attr_storage_type, seed, device):
    adj_per_eps = []
    attr_per_eps = []

    adversary = create_attack(attack, binary_attr, attr, adj=adj, labels=labels,
                              model=model, idx_attack=idx_attack, device=device, **attack_params)

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
    return adj_per_eps, attr_per_eps


def evaluate_global_attack(models_and_hyperparams, labels, epsilons, adj_per_eps, attr_per_eps, seed, device, idx):
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

                try:
                    pred_logits_target = model(attr_perturbed.to(device), adj_perturbed.to(device))
                    acc_test_target = accuracy(pred_logits_target.cpu(), labels.cpu(), idx)
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
    return results
