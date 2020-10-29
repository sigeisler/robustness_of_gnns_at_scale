
import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment
import seml
import torch

from rgnn import data
from rgnn.io import Storage
from rgnn.models import create_model
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
    dataset = 'cora_ml'
    model_params = {
        'label': 'Vanilla GCN',
        'model': 'GCN',
        'dropout': 0.5,
        'n_filters': 64,
        'gdc_params': None,
        'svd_params': None,
        'jaccard_params': None,
        'do_cache_adj_prep': True
    }
    train_params = {
        'lr': 1e-2,
        'weight_decay': 5e-4,
        'patience': 300,
        'max_epochs': 3000
    }
    binary_attr = False
    seed = 0
    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained'
    device = 0
    display_steps = 10


@ex.automain
def run(dataset: str, model_params: Dict[str, Any], train_params: Dict[str, Any], binary_attr: bool,
        seed: int, artifact_dir: str, model_storage_type: str, device: Union[str, int], display_steps: int):
    logging.info({
        'dataset': dataset, 'model_params': model_params, 'train_params': train_params, 'binary_attr': binary_attr,
        'seed': seed, 'artifact_dir': artifact_dir, 'model_storage_type': model_storage_type, 'device': device,
        'display_steps': display_steps
    })

    torch.manual_seed(seed)
    np.random.seed(seed)

    attr, adj, labels = data.prep_graph(dataset, device=device, binary_attr=binary_attr)
    n_features = attr.shape[1]
    n_classes = int(labels.max() + 1)

    idx_train, idx_val, idx_test = data.split(labels.cpu().numpy())

    # Collect all hyperparameters of model
    hyperparams = dict(model_params)
    hyperparams.update({
        'n_features': n_features,
        'n_classes': n_classes
    })

    model = create_model(hyperparams).to(device)
    if hasattr(model, 'fit'):
        model.fit(adj, attr, labels=labels, idx_train=idx_train,
                  idx_val=idx_val, display_step=display_steps, **train_params)
        trace_val, trace_train = None, None
    else:
        trace_val, trace_train = train(model=model, attr=attr, adj=adj, labels=labels, idx_train=idx_train,
                                       idx_val=idx_val, display_step=display_steps, **train_params)

    model.eval()
    prediction = model(attr, adj)
    test_accuracy = accuracy(prediction, labels, idx_test)
    logging.info(f'Test accuracy is {test_accuracy} with seed {seed}')

    storage = Storage(artifact_dir)
    params = dict(dataset=dataset, binary_attr=binary_attr, seed=seed, **hyperparams)
    model_path = storage.save_model(model_storage_type, params, model)

    return {
        'accuracy': test_accuracy,
        'trace_val': trace_val,
        'trace_train': trace_train,
        'model_path': model_path
    }
