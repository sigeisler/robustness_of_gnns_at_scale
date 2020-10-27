import logging
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from rgnn.fgsm import FGSM
from rgnn.pgd import PGD
from rgnn.models import DenseGCN, GCN, RGNN, RGCN
from rgnn import data
from rgnn.data import SparseGraph


def train(model, attr, adj, labels, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, display_step=50):
    """Train a model using either standard or adversarial training.
    Parameters
    ----------
    model: torch.nn.Module
        Model which we want to train.
    attr: torch.Tensor [n, d]
        Dense attribute matrix.
    adj: torch.Tensor [n, n]
        Dense adjacency matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: array-like [?]
        Indices of the training nodes.
    idx_val: array-like [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    Returns
    -------
    train_val, trace_val: list
        A tupole of lists of values of the validation loss during training.
    """
    trace_train = []
    trace_val = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf

    model.train()
    for it in tqdm(range(max_epochs), desc='Training...'):
        optimizer.zero_grad()

        logits = model(attr, adj)
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss_val = F.cross_entropy(logits[idx_val], labels[idx_val])

        loss_train.backward()
        optimizer.step()
        trace_train.append(loss_train.detach().item())
        trace_val.append(loss_val.detach().item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if it % display_step == 0:
            logging.info(f'\nEpoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} ')

    # restore the best validation state
    model.load_state_dict(best_state)
    return trace_val, trace_train


def train_and_attack(graph: SparseGraph, epsilons: List[float], model_attack: Dict[str, Any], attr: torch.Tensor,
                     adj: torch.Tensor, labels: torch.Tensor, idx_train: torch.Tensor, idx_val: torch.Tensor,
                     idx_test: torch.Tensor, attr_per_eps: List[torch.Tensor], adj_per_eps: List[torch.Tensor],
                     device: Union[int, torch.device] = 0) -> Tuple[float, Dict[str, Any]]:
    """Train network and attack via the FGSM-like attack on the surrogate Model. Thereafter, the perturbed adjacency
    matrices are used for predictions on the target model (trained on clean graph).

    Parameters
    ----------
    graph: SparseGraph
        Raw graph (unnormalized without self-loops)
    epsilons : List[float]
        The fraction of edges to be changed
    model_attack : Dict[str, Any]
        Parameter for the model to be attacked
    attr : torch.Tensor
        Dense [n,d] tensor with the attributes
    adj : torch.Tensor
        Dense [n,d] tensor with the attributes
    labels : torch.Tensor
        Dense [n] tensor with the labels
    idx_test : torch.Tensor
        Dense [?] tensor with the indices for the test labels
    attr_per_eps : List[torch.Tensor]
        List of perturbed attributes (CPU)
    adj_per_eps : List[torch.Tensor]
        List of perturbed attributes (CPU)
    device : Union[int, torch.device], optional
        `cpu` or GPU id, by default 0

    Returns
    -------
    Dict[str, Any]
       Results for the target model

    Raises
    ------
    NotImplementedError
        If the target model in unknown
    """
    n_vertices, n_features = graph.attr_matrix.shape
    n_classes = graph.labels.max() + 1

    if model_attack['model'] == 'GCN':
        model = GCN(
            n_features=n_features,
            n_classes=n_classes,
            **model_attack
        ).to(device)
    elif model_attack['model'] == 'RGNN':
        model = RGNN(
            n_features=n_features,
            n_classes=n_classes,
            **model_attack
        ).to(device)
    elif model_attack['model'] == 'RGCN':
        model = RGCN(
            nnodes=n_vertices,
            nfeat=n_features,
            nhid=model_attack['n_filters'],
            nclass=n_classes
        ).to(device)
    else:
        raise NotImplementedError()

    if isinstance(model, RGCN):
        model.fit(edge_idx=adj.indices(), attr_idx=attr.to_sparse()._indices(),
                  labels=labels, n=n_vertices, d=n_features, nc=n_classes,
                  idx_train=idx_train, idx_val=idx_val, max_epochs=model_attack['train_params']['max_epochs'])
    else:
        trace_val, _ = train(model=model, attr=attr, adj=adj, labels=labels,
                             idx_train=idx_train, idx_val=idx_val, **model_attack['train_params'])
        print(trace_val)
    model.eval()

    results = []
    for epsilon, attr_adversary, adj_adversary in zip(epsilons, attr_per_eps, adj_per_eps):
        pred_logits_pert = model(attr_adversary.to(device), adj_adversary.to(device))
        acc_test = (pred_logits_pert.argmax(1)[idx_test] == labels[idx_test]).sum().item() / len(idx_test)
        logging.info(f'Test accuracy {acc_test} at {epsilon} (#edges:{adj_adversary._nnz()})')
        results.append({'model': model_attack['label'], 'epsilon': epsilon, 'accuracy': acc_test})

    return results
