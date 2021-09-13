
import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment

import torch

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.models import create_model
from rgnn_at_scale.train import train
from rgnn_at_scale.helper.utils import accuracy
from rgnn_at_scale.helper import utils
import torch.nn.functional as F

try:
    import seml
except:  # noqa: E722
    seml = None


ex = Experiment()

if seml is not None:
    seml.setup_logger(ex)


@ex.config
def config():
    overwrite = None

    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    # default params
    data_dir = './datasets'
    dataset = 'ogbn-products'
    make_undirected = True
    binary_attr = False
    data_device = 0

    device = 0
    seed = 0

    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained'
    model_params = dict(
        label="Vanilla SGC",
        model="SGC",
        K=4,
        cached=True
    )

    train_params = dict(
        lr=1e-2,
        weight_decay=0,
        patience=300,
        max_epochs=3000
    )

    display_steps = 100
    debug_level = "info"


@ex.automain
def run(data_dir: str, dataset: str, model_params: Dict[str, Any], train_params: Dict[str, Any], binary_attr: bool,
        make_undirected: bool, seed: int, artifact_dir: str, model_storage_type: str,
        device: Union[str, int], data_device: Union[str, int], display_steps: int, debug_level: str):
    """
    Instantiates a sacred experiment executing a training run for a given model configuration.
    Saves the model to storage and evaluates its accuracy. 

    Parameters
    ----------
    data_dir : str
        Path to data folder that contains the dataset
    dataset : str
        Name of the dataset. Either one of: `cora_ml`, `citeseer`, `pubmed` or an ogbn dataset
    model_params : Dict[str, Any], optional
        The hyperparameters of the model to be passed as keyword arguments to the constructor of the model class.
        This dict must contain the key "model" specificing the model class. Supported model classes are:
            - GCN
            - DenseGCN
            - RGCN
    train_params : Dict[str, Any], optional
        The training/hyperparamters to be passed as keyword arguments to the model's ".fit()" method or to 
        the global "train" method if "model.fit()" is undefined.
    device : Union[int, torch.device]
        The device to use for training. Must be `cpu` or GPU id
    data_device : Union[int, torch.device]
        The device to use for storing the dataset.
        In all other cases device takes precedence over data_device
    make_undirected : bool
        Normalizes adjacency matrix with symmetric degree normalization (non-scalable implementation!)
    binary_attr : bool
        If true the attributes are binarized (!=0)
    artifact_dir: str
        The path to the folder that acts as TinyDB Storage for trained models
    model_storage_type: str
        The name of the storage (TinyDB) table name the model is stored into.

    Returns
    -------
    Dict[str, any]
        A dictionary with the test set accuracy, the training & validation loss as well as the path to the trained model. 
    """
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

    logging.info({
        'dataset': dataset, 'model_params': model_params, 'train_params': train_params, 'binary_attr': binary_attr,
        'make_undirected': make_undirected, 'seed': seed, 'artifact_dir': artifact_dir,
        'model_storage_type': model_storage_type, 'device': device,
        'display_steps': display_steps, 'data_device': data_device
    })

    torch.manual_seed(seed)
    np.random.seed(seed)
    graph = prep_graph(dataset, data_device, dataset_root=data_dir, make_undirected=make_undirected,
                       binary_attr=binary_attr, return_original_split=dataset.startswith('ogbn'))

    attr, adj, labels = graph[:3]
    if len(graph) == 3 or graph[3] is None:  # TODO: This is weird
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    n_features = attr.shape[1]
    n_classes = int(labels[~labels.isnan()].max() + 1)

    logging.info(f"Training set size: {len(idx_train)}")
    logging.info(f"Validation set size: {len(idx_val)}")
    logging.info(f"Test set size: {len(idx_test)}")

    # Collect all hyperparameters of model
    ppr_cache = None
    hyperparams = dict(model_params)
    hyperparams.update({
        'n_features': n_features,
        'n_classes': n_classes,
        'train_params': train_params
    })

    model = create_model(hyperparams).to(device)

    logging.info("Memory Usage after loading the dataset:")
    logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

    if hasattr(model, 'fit'):
        trace = model.fit(adj, attr,
                          labels=labels,
                          idx_train=idx_train,
                          idx_val=idx_val,
                          display_step=display_steps,
                          dataset=dataset,
                          make_undirected=make_undirected,
                          **train_params)

        trace_val, trace_train = trace if trace is not None else (None, None)

    else:
        trace_val, trace_train, _, _ = train(
            model=model, attr=attr.to(device), adj=adj.to(device), labels=labels.to(device),
            idx_train=idx_train, idx_val=idx_val, display_step=display_steps, **train_params)

    model.eval()

    with torch.no_grad():
        model.eval()
        prediction = model(attr, adj)
        test_accuracy = accuracy(prediction.cpu(), labels.cpu(), idx_test)

    logging.info(f'Test accuracy is {test_accuracy} with seed {seed}')

    storage = Storage(artifact_dir, experiment=ex)
    params = dict(dataset=dataset, binary_attr=binary_attr, make_undirected=make_undirected,
                  seed=seed, **hyperparams)

    model_path = storage.save_model(model_storage_type, params, model)

    return {
        'accuracy': test_accuracy,
        'trace_val': trace_val,
        'trace_train': trace_train,
        'model_path': model_path
    }
