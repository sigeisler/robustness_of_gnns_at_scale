"""Training code.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from rgnn_at_scale.helper.utils import accuracy


def train(model, attr, adj, labels, idx_train, idx_val,
          lr, weight_decay, patience, max_epochs, display_step=50):
    """Train a model using either standard training.
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
    trace_loss_train = []
    trace_loss_val = []
    trace_acc_train = []
    trace_acc_val = []
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

        trace_loss_train.append(loss_train.detach().item())
        trace_loss_val.append(loss_val.detach().item())

        train_acc = accuracy(logits, labels, idx_train)
        val_acc = accuracy(logits, labels, idx_val)

        trace_acc_train.append(train_acc)
        trace_acc_val.append(val_acc)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if it % display_step == 0:
            logging.info(f'\nEpoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f}, '
                         f'acc_train: {train_acc:.5f}, acc_val: {val_acc:.5f} ')

    # restore the best validation state
    model.load_state_dict(best_state)
    return trace_loss_val, trace_loss_train, trace_acc_val, trace_acc_train
