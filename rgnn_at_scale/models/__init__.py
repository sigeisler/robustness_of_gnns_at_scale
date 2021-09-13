from typing import Any, Dict, Union

from rgnn_at_scale.models.gcn import GCN, DenseGCN
from rgnn_at_scale.models.sgc import SGC
from rgnn_at_scale.models.rgnn import RGNN
from rgnn_at_scale.models.rgcn import RGCN


MODEL_TYPE = Union[SGC, GCN, RGNN, RGCN]


def create_model(hyperparams: Dict[str, Any]) -> MODEL_TYPE:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    hyperparams : Dict[str, Any]
        Containing the hyperparameters.

    Returns
    -------
    model: MODEL_TYPE
        The created instance.
    """
    if 'model' not in hyperparams or hyperparams['model'] == 'GCN':
        return GCN(**hyperparams)
    if hyperparams['model'] == "SGC":
        return SGC(**hyperparams)
    if hyperparams['model'] == 'DenseGCN':
        return DenseGCN(**hyperparams)
    if hyperparams['model'] == 'RGCN':
        return RGCN(**hyperparams)
    return RGNN(**hyperparams)


__all__ = [GCN,
           DenseGCN,
           RGNN,
           RGCN,
           create_model,
           MODEL_TYPE]
