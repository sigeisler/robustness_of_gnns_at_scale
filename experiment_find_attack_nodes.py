import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
from sacred import Experiment
import seml
import torch
from torch.nn import functional as F

from rgnn_at_scale.models import BATCHED_PPR_MODELS
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
    seed = 0
    artifact_dir = 'cache_debug'
    model_storage_type = 'pretrained'
    binary_attr = False
    device = "cpu"
    model_label = 'Vanilla PPRGo'
    make_undirected = True
    make_unweighted = True
    data_dir = './datasets'
    data_device = 'cpu'
    topk = 10


@ex.automain
def run(data_dir: str, dataset: str, binary_attr: bool, make_undirected: bool, make_unweighted: bool, seed: int,
        artifact_dir: str, model_label: str, model_storage_type: str, device: Union[str, int], topk: int,
        data_device: Union[str, int]):
    logging.info({
        'dataset': dataset, 'binary_attr': binary_attr, 'seed': seed, 'device': device,
        'artifact_dir': artifact_dir, 'model_label': model_label, 'model_storage_type': model_storage_type
    })

    results = []
    graph = prep_graph(dataset, data_device, dataset_root=data_dir,
                       make_undirected=make_undirected,
                       make_unweighted=make_unweighted,
                       binary_attr=binary_attr,
                       return_original_split=dataset.startswith('ogbn'))
    attr, adj, labels = graph[:3]
    if len(graph) == 3:
        idx_train, idx_val, idx_test = split(labels.cpu().numpy())
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']

    storage = Storage(artifact_dir, experiment=ex)

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr,
                        seed=seed)

    if model_label is not None and model_label:
        model_params['label'] = model_label
    models_and_hyperparams = storage.find_models(model_storage_type, model_params)

    for model, hyperparams in models_and_hyperparams:
        model = model.to(device)
        model.eval()

        if type(model) in BATCHED_PPR_MODELS.__args__:
            log_prob = F.log_softmax(model.forward(attr.to(device), adj.to(device), ppr_idx=idx_test), dim=-1)
        else:
            log_prob = model(data=attr.to(device), adj=adj.to(device))[idx_test]
            if model.do_omit_softmax:
                log_prob = F.log_softmax(log_prob)

        _, max_confidence_nodes_idx = torch.topk(log_prob.max(-1).values, k=topk)
        _, min_confidence_nodes_idx = torch.topk(-log_prob.max(-1).values, k=topk)
        rand_nodes_idx = torch.randint(idx_test.shape[0], (1, topk * 2))
        results.append({
            'hyperparams': hyperparams,
            'log_prob': log_prob.cpu(),
            'max_confidence_nodes_idx': idx_test[max_confidence_nodes_idx],
            'min_confidence_nodes_idx': idx_test[min_confidence_nodes_idx],
            'rand_confidence_nodes_idx': idx_test[rand_nodes_idx],
        })

    return {
        'results': results
    }
