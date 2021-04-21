
import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment
import seml
import torch

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.models import create_model, PPRGoWrapperBase
from rgnn_at_scale.train import train
from rgnn_at_scale.helper.utils import accuracy
from rgnn_at_scale.helper import utils
import torch.nn.functional as F

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
        'label': 'Vanilla PPRGo Diffusion Embedding',
        'model': 'DenseGCN',
        'dropout': 0.5,
        'n_filters': 64,
        'hidden_size': 64,
        'nlayers': 3,
        'ppr_normalization': 'row',
        'topk': 64, 'alpha': 0.1, 'eps': 1e-03,
        'skip_connection': True,
        'gdc_params': None,
        'svd_params': None,
        'batch_norm': False,
        'jaccard_params': None,
        'do_cache_adj_prep': True
    }
    train_params = {
        'lr': 1e-2,
        'weight_decay': 5e-4,
        'patience': 300,
        'max_epochs': 2
    }
    binary_attr = False
    normalize = False
    normalize_attr = False
    make_undirected = True
    make_unweighted = True
    seed = 1
    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained'
    ppr_cache_params = None
    ppr_cache_params = dict(
        data_artifact_dir="/nfs/students/schmidtt/cache",
        data_storage_type="ppr"
    )
    device = 'cpu'
    display_steps = 10
    data_dir = './datasets'
    data_device = 'cpu'


@ex.automain
def run(data_dir: str, dataset: str, model_params: Dict[str, Any], train_params: Dict[str, Any], binary_attr: bool,
        make_undirected: bool, make_unweighted: bool, normalize: bool, normalize_attr: str, seed: int, artifact_dir: str,
        model_storage_type: str, ppr_cache_params: Dict[str, str], device: Union[str, int], data_device: Union[str, int], display_steps: int):
    """
    Instantiates a SEML experiment executing a training run for a given model configuration.
    Saves the model to storage and evaluates its accuracy. 

    Parameters
    ----------
    data_dir : str, optional
        Path to data folder that contains the dataset
    dataset : str
        Name of the dataset. Either one of: `cora_ml`, `citeseer`, `pubmed` or an ogbn dataset
    model_params : Dict[str, Any]
        The hyperparameters of the model to be passed as keyword arguments to the constructor of the model class.
        This dict must contain the key "label" specificing the model class.
    train_params : Dict[str, Any]

    device : Union[int, torch.device]
        `cpu` or GPU id, by default 0
    normalize : bool, optional
        Normalize adjacency matrix with symmetric degree normalization (non-scalable implementation!), by default False
    binary_attr : bool, optional
        If true the attributes are binarized (!=0), by default False

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        dense attribute tensor, sparse adjacency matrix (normalized) and labels tensor.
    """

    logging.info({
        'dataset': dataset, 'model_params': model_params, 'train_params': train_params, 'binary_attr': binary_attr,
        'make_undirected': make_undirected, 'make_unweighted': make_unweighted, 'normalize': normalize, 'normalize_attr': normalize_attr,
        'seed': seed, 'artifact_dir': artifact_dir, 'model_storage_type': model_storage_type, 'ppr_cache_params': ppr_cache_params,
        'device': device, 'display_steps': display_steps
    })

    torch.manual_seed(seed)
    np.random.seed(seed)
    breakpoint()
    logging.info("there should be a breakpoint here")
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
    n_classes = int(labels[~labels.isnan()].max() + 1)

    print("Training set size: ", len(idx_train))
    print("Validation set size: ", len(idx_val))
    print("Test set size: ", len(idx_test))

    # Collect all hyperparameters of model
    ppr_cache = None
    if ppr_cache_params is not None:
        ppr_cache = dict(ppr_cache_params)
        ppr_cache.update(dict(
            dataset=dataset,
            normalize=normalize,
            make_undirected=make_undirected,
            make_unweighted=make_unweighted,
        ))
    hyperparams = dict(model_params)
    hyperparams.update({
        'n_features': n_features,
        'n_classes': n_classes,
        'ppr_cache_params': ppr_cache
    })

    model = create_model(hyperparams).to(device)

    logging.debug("Memory Usage after loading the dataset:")
    logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

    if hasattr(model, 'fit'):
        trace = model.fit(adj, attr,
                          labels=labels,
                          idx_train=idx_train,
                          idx_val=idx_val,
                          display_step=display_steps,
                          dataset=dataset,
                          normalize=normalize,
                          make_undirected=make_undirected,
                          make_unweighted=make_unweighted,
                          **train_params)

        trace_val, trace_train = trace if trace is not None else (None, None)

    else:
        trace_val, trace_train = train(model=model, attr=attr, adj=adj, labels=labels, idx_train=idx_train,
                                       idx_val=idx_val, display_step=display_steps, **train_params)

    model.eval()

    # For really large graphs we don't want to compute predictions for all nodes, just the test nodes is enough.
    # Calculating predictions for a sub-set of nodes is only possible for batched gnns like PPRGo
    with torch.no_grad():
        model.eval()
        if isinstance(model, PPRGoWrapperBase):
            prediction = model(attr, adj, ppr_idx=idx_test)
            test_accuracy = (prediction.cpu().argmax(1) == labels.cpu()[idx_test]).float().mean().item()
        else:
            prediction = model(attr, adj)
            test_accuracy = accuracy(prediction.cpu(), labels.cpu(), idx_test)

    logging.info(f'Test accuracy is {test_accuracy} with seed {seed}')

    storage = Storage(artifact_dir, experiment=ex)
    params = dict(dataset=dataset,
                  binary_attr=binary_attr,
                  normalize=normalize,
                  normalize_attr=normalize_attr,
                  make_undirected=make_undirected,
                  make_unweighted=make_unweighted,
                  seed=seed, **hyperparams)

    model_path = storage.save_model(model_storage_type, params, model)

    return {
        'accuracy': test_accuracy,
        'trace_val': trace_val,
        'trace_train': trace_train,
        'model_path': model_path
    }
