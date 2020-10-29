import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment
import seml
import torch

from sparse_smoothing.prediction import predict_smooth_gnn
from sparse_smoothing.cert import binary_certificate

from rgnn.data import prep_graph, split
from rgnn.io import Storage


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
    sample_params = {
        'n_samples': 10_000,
        'pf_plus_adj': 0.001,
        'pf_plus_att': 0,
        'pf_minus_adj': 0.4,
        'pf_minus_att': 0
    }
    n_samples_pre_eval = 100
    conf_alpha = 0.05
    seed = 0
    batch_size = 1
    artifact_dir = 'cache_debug'
    smoothing_result_storage_type = 'smoothing'
    model_storage_type = 'pretrained'
    device = 0


def calc_certification_ratio(smoothing_result: Dict[str, Any], idx_selected: np.ndarray, labels: np.ndarray,
                             mask: np.ndarray = None) -> np.ndarray:
    """Calculation of the certification ratio. `R(r_a, r_d)` in our paper.

    Parameters
    ----------
    smoothing_result : Dict[str, Any]
        Dictionary with smoothing results.
    idx_selected : np.ndarray
        Array containing the indices of e.g. the test nodes.
    labels : np.ndarray, optional
        Ground truth class labels.
    mask : np.ndarray, optional
        To select only a subset of nodes e.g. by degree, by default None.

    Returns
    -------
    np.ndarray
        Bivariate certification ratio R(r_a, r_d).
    """
    grid_lower = smoothing_result['grid_lower'][idx_selected]
    grid_upper = smoothing_result['grid_upper'][idx_selected]
    if mask is not None:
        grid_lower = grid_lower[mask[idx_selected]]
        grid_upper = grid_upper[mask[idx_selected]]

    correctly_classified = (smoothing_result['votes'][idx_selected].argmax(1) == labels[idx_selected])
    if mask is not None:
        correctly_classified = correctly_classified[mask[idx_selected]]
    heatmap_loup = (
        (grid_lower > grid_upper)
        & np.tile(correctly_classified, [grid_lower.shape[-1], grid_lower.shape[-2], 1]).T
    )

    heatmap_loup = heatmap_loup.mean(0).T
    heatmap_loup[0, 0] = correctly_classified.mean()

    return heatmap_loup


@ex.automain
def run(dataset: str, sample_params: Dict[str, Any], n_samples_pre_eval: int, conf_alpha: float,
        seed: int, batch_size: int, artifact_dir: str, smoothing_result_storage_type: str,
        model_storage_type: str, device: Union[str, int]):
    logging.info({
        'dataset': dataset, 'sample_params': sample_params, 'n_samples_pre_eval': n_samples_pre_eval,
        'conf_alpha': conf_alpha, 'seed': seed, 'artifact_dir': artifact_dir, 'model_storage_type': model_storage_type,
        'smoothing_result_storage_type': smoothing_result_storage_type, 'device': device, 'batch_size': batch_size
    })

    binary_attr = True

    torch.manual_seed(seed)
    np.random.seed(seed)

    attr, adj, labels = prep_graph(dataset, device=device, binary_attr=binary_attr)
    n_nodes, n_features = attr.shape
    n_classes = int(labels.max() + 1)

    idx_train, idx_val, idx_test = split(labels.cpu().numpy())

    model_params = dict(dataset=dataset, binary_attr=binary_attr, seed=seed)
    smoothing_params = dict(model_params)
    smoothing_params['sample_params'] = sample_params

    storage = Storage(artifact_dir)
    smoothing_results = storage.find_artifacts(smoothing_result_storage_type, smoothing_params)
    smoothing_results = {
        result['params']['model_id']: result['artifact']
        for result
        in smoothing_results
    }
    model_hyperparams_and_id = storage.find_models(model_storage_type, model_params, return_model_id=True)

    for model, hyperparams, id in model_hyperparams_and_id:
        attr_idx = attr.to_sparse().indices()
        if id not in smoothing_results:
            model = model.to(device)

            sample_params_pe = sample_params.copy()
            sample_params_pe['n_samples'] = n_samples_pre_eval
            pre_votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=adj.indices(), sample_config=sample_params_pe,
                                           model=model, n=n_nodes, d=n_features, nc=n_classes, batch_size=batch_size)

            votes = predict_smooth_gnn(attr_idx=attr_idx, edge_idx=adj.indices(), sample_config=sample_params,
                                       model=model, n=n_nodes, d=n_features, nc=n_classes, batch_size=batch_size)

            # we are perturbing ONLY the ATTRIBUTES
            if sample_params['pf_plus_adj'] == 0 and sample_params['pf_minus_adj'] == 0:
                grid_base, grid_lower, grid_upper = binary_certificate(
                    votes=votes, pre_votes=pre_votes, n_samples=sample_params['n_samples'], conf_alpha=conf_alpha,
                    pf_plus=sample_params['pf_plus_att'], pf_minus=sample_params['pf_minus_att'])
            # we are perturbing ONLY the GRAPH
            elif sample_params['pf_plus_att'] == 0 and sample_params['pf_minus_att'] == 0:
                grid_base, grid_lower, grid_upper = binary_certificate(
                    votes=votes, pre_votes=pre_votes, n_samples=sample_params['n_samples'], conf_alpha=conf_alpha,
                    pf_plus=sample_params['pf_plus_adj'], pf_minus=sample_params['pf_minus_adj'])
            else:
                raise NotImplementedError('Please only perturb either the attributes or the structure!')

            smoothing_result = {
                'grid_base': grid_base,
                'grid_lower': grid_lower,
                'grid_upper': grid_upper,
                'votes': votes,
                'pre_votes': pre_votes
            }
            smoothing_results[id] = smoothing_result
            storage.save_artifact(
                smoothing_result_storage_type, dict(model_id=id, **smoothing_params), smoothing_result
            )

    results = []
    for model, hyperparams, id in model_hyperparams_and_id:
        cert_ratio = calc_certification_ratio(smoothing_results[id], idx_test, labels.cpu().numpy())
        accum_certs = cert_ratio.sum() - cert_ratio[0][0]
        results.append(dict(
            label=hyperparams['label'],
            accum_certs=accum_certs
        ))

    return {
        'results': results
    }
