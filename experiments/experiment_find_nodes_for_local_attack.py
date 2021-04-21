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
from experiments.common import sample_attack_nodes


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
    artifact_dir = 'cache'
    model_storage_type = 'victim_cora'
    model_label = 'Vanilla PPRGo'

    topk = 10

    device = "cpu"
    data_device = 'cpu'

    data_dir = './datasets'
    binary_attr = False
    normalize = False
    normalize_attr = False
    make_undirected = True
    make_unweighted = True


@ex.automain
def run(data_dir: str, dataset: str, binary_attr: bool, make_undirected: bool, make_unweighted: bool, normalize: bool,
        normalize_attr: str, seed: int, artifact_dir: str, model_label: str, model_storage_type: str, topk: int,
        device: Union[str, int], data_device: Union[str, int]):

    results = []
    # To increase consistency between runs
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    model_params = dict(dataset=dataset,
                        binary_attr=binary_attr,
                        normalize=normalize,
                        normalize_attr=normalize_attr,
                        make_undirected=make_undirected,
                        make_unweighted=make_unweighted,
                        seed=seed)

    if model_label is not None and model_label:
        model_params["label"] = model_label

    storage = Storage(artifact_dir, experiment=ex)
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

        max_confidence_nodes_idx, min_confidence_nodes_idx, rand_nodes_idx = sample_attack_nodes(log_prob, labels[idx_test], topk, idx_test)
        
        results.append({
            'hyperparams': hyperparams,
            'log_prob': log_prob.cpu(),
            'max_confidence_nodes_idx': max_confidence_nodes_idx,
            'min_confidence_nodes_idx': min_confidence_nodes_idx,
            'rand_confidence_nodes_idx': rand_nodes_idx
        })

    return {
        'results': results
    }
