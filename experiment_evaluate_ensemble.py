import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
from sacred import Experiment
import seml
import torch

from rgnn_at_scale.data import prep_graph, split
from rgnn_at_scale.models import PPRGoWrapperBase
from rgnn_at_scale.io import Storage
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
    data_dir = "/nfs/students/schmidtt/datasets/"
    dataset = 'cora_ml'  # Options are 'cora_ml' and 'citeseer' (or with a big GPU 'pubmed')
    binary_attr = False
    make_undirected = True
    make_unweighted = True
    artifact_dir = 'cache'
    model_storage_type = 'attack_cora'
    device = "cpu"
    data_device = "cpu"
    model_label = None
    forward_batch_size = 128


@ex.automain
def run(data_dir: str, dataset: str,  binary_attr: bool, make_undirected: bool, make_unweighted: bool, normalize: bool, normalize_attr: str, seed: int,
        artifact_dir: str,  model_label: str, model_storage_type: str, device: Union[str, int], data_device: Union[str, int], forward_batch_size: int):
    logging.info({
        'dataset': dataset, 'binary_attr': binary_attr, 'seed': seed,
        'normalize': normalize, 'normalize_attr': normalize_attr,
        'make_undirected': make_undirected, 'make_unweighted': make_unweighted,
        'artifact_dir': artifact_dir, 'model_label': model_label,
        'model_storage_type': model_storage_type, 'device': device, 'forward_batch_size': forward_batch_size
    })

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
    n_classes = int(labels[~labels.isnan()].max() + 1)

    storage = Storage(artifact_dir, experiment=ex)

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr)

    if model_label is not None and model_label:
        model_params['label'] = model_label
    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    with torch.no_grad():
        pred_logits = torch.zeros((len(models_and_hyperparams), idx_test.shape[0], n_classes))
        for i, (model, hyperparams) in enumerate(models_and_hyperparams):
            model = model.to(device)
            model.eval()

            torch.manual_seed(hyperparams["seed"])
            np.random.seed(hyperparams["seed"])

            if isinstance(model, PPRGoWrapperBase):
                model.forward_batch_size = forward_batch_size
                pred_logits[i] = model(attr, adj, ppr_idx=idx_test)

            else:
                pred_logits[i] = model(attr, adj)[idx_test]
                test_accuracy = accuracy(prediction.cpu(), labels.cpu(), idx_test)

        prediction = pred_logits.mean(0)
        test_accuracy_models = (pred_logits.cpu().argmax(-1) == labels.cpu()
                                [idx_test].view(1, -1).repeat(len(models_and_hyperparams), 1)).float().mean(-1)
        test_accuracy_ensemble = (prediction.cpu().argmax(1) == labels.cpu()[idx_test]).float().mean().item()

        results = {
            'prediction_models': pred_logits.detach().cpu(),
            'prediction': prediction.detach().cpu(),
            'test_accuracy_models': test_accuracy_models.detach().cpu(),
            'test_accuracy_ensemble': test_accuracy_ensemble
        }

    return {
        'results': results
    }
