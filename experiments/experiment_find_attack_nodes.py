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
from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.models import DenseGCN, GCN
from rgnn_at_scale.train import train
from rgnn_at_scale.helper.utils import accuracy


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
    dataset = 'citeseer'  # Options are 'cora_ml' and 'citeseer' (or with a big GPU 'pubmed')
    seed = 0
    artifact_dir = 'cache'
    model_storage_type = 'nettack_citeseer'
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

        with torch.no_grad():
            if type(model) in BATCHED_PPR_MODELS.__args__:
                log_prob = F.log_softmax(model.forward(attr, adj, ppr_idx=idx_test), dim=-1).detach().cpu()
            else:
                log_prob = model(data=attr.to(device), adj=adj.to(device))[idx_test].detach().cpu()
                if model.do_omit_softmax:
                    log_prob = F.log_softmax(log_prob)

        labels = labels.cpu()
        correctly_classifed = log_prob.max(-1).indices == labels[idx_test]
        _, max_confidence_nodes_idx = torch.topk(log_prob[correctly_classifed].max(-1).values, k=topk)
        _, min_confidence_nodes_idx = torch.topk(-log_prob[correctly_classifed].max(-1).values, k=topk)
        rand_nodes_idx = torch.randint(correctly_classifed.sum(), (1, topk * 3))
        results.append({
            'hyperparams': hyperparams,
            'log_prob': log_prob.cpu(),
            'max_confidence_nodes_idx': idx_test[correctly_classifed][max_confidence_nodes_idx],
            'min_confidence_nodes_idx': idx_test[correctly_classifed][min_confidence_nodes_idx],
            'rand_confidence_nodes_idx': idx_test[correctly_classifed][rand_nodes_idx].flatten()
        })

    return {
        'results': results
    }
