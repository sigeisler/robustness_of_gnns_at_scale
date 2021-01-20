
import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment
import seml
import torch

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.io import Storage
from rgnn_at_scale.models import create_model, PPRGoWrapperBase
from rgnn_at_scale.train import train
from rgnn_at_scale.utils import accuracy
import torch.nn.functional as F
from pprgo import utils as ppr_utils

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
def run(data_dir: str, dataset: str, model_params: Dict[str, Any], train_params: Dict[str, Any], binary_attr: bool, seed: int,
        artifact_dir: str, model_storage_type: str, device: Union[str, int], data_device: Union[str, int], display_steps: int):
    logging.info({
        'dataset': dataset, 'model_params': model_params, 'train_params': train_params, 'binary_attr': binary_attr,
        'seed': seed, 'artifact_dir': artifact_dir, 'model_storage_type': model_storage_type, 'device': device,
        'display_steps': display_steps
    })

    torch.manual_seed(seed)
    np.random.seed(seed)

    graph = prep_graph(dataset, data_device, dataset_root=data_dir, binary_attr=binary_attr,
                       return_original_split=dataset.startswith('ogbn'))

    logging.info("prep_graph")
    logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

    attr, adj, labels = graph[:3]
    if len(graph) == 3:
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    logging.info("idx_train")
    logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

    n_features = attr.shape[1]
    n_classes = int(labels[~labels.isnan()].max() + 1)

    print("Training set size: ", len(idx_train))
    print("Validation set size: ", len(idx_val))
    print("Test set size: ", len(idx_test))

    # Collect all hyperparameters of model
    hyperparams = dict(model_params)
    hyperparams.update({
        'n_features': n_features,
        'n_classes': n_classes
    })

    model = create_model(hyperparams).to(device)

    logging.info("model")
    logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

    if hasattr(model, 'fit'):
        trace = model.fit(adj, attr, labels=labels, idx_train=idx_train,
                          idx_val=idx_val, display_step=display_steps, **train_params)
        if trace is None:
            trace_val, trace_train = None, None
        else:
            trace_val, trace_train = trace

    else:
        trace_val, trace_train = train(model=model, attr=attr, adj=adj, labels=labels, idx_train=idx_train,
                                       idx_val=idx_val, display_step=display_steps, **train_params)

    model.eval()

    # For really large graphs we don't want to compute predictions for all nodes, just the test nodes is enough.
    if isinstance(model, PPRGoWrapperBase):
        prediction = model(attr, adj, ppr_idx=idx_test)
        test_accuracy = (prediction.cpu().argmax(1) == labels.cpu()[idx_test]).float().mean().item()
    else:
        prediction = model(attr, adj)
        test_accuracy = accuracy(prediction.cpu(), labels.cpu(), idx_test)

    logging.info(f'Test accuracy is {test_accuracy} with seed {seed}')

    storage = Storage(artifact_dir, experiment=ex)
    params = dict(dataset=dataset, binary_attr=binary_attr, seed=seed, **hyperparams)
    model_path = storage.save_model(model_storage_type, params, model)

    return {
        'accuracy': test_accuracy,
        'trace_val': trace_val,
        'trace_train': trace_train,
        'model_path': model_path
    }
