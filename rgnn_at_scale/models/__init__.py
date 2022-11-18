from typing import Any, Dict, Union

from rgnn_at_scale.models.gcn import GCN, DenseGCN
from rgnn_at_scale.models.sgc import SGC
from rgnn_at_scale.models.rgnn import RGNN
from rgnn_at_scale.models.rgcn import RGCN
from rgnn_at_scale.models.pprgo import (PPRGoWrapperBase, RobustPPRGoWrapper, PPRGoWrapper)


MODEL_TYPE = Union[SGC, GCN, DenseGCN, RGNN, RGCN, RobustPPRGoWrapper, PPRGoWrapper]
BATCHED_PPR_MODELS = Union[RobustPPRGoWrapper, PPRGoWrapper]


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
    if hyperparams['model'] == "RobustPPRGo":
        return RobustPPRGoWrapper(**hyperparams)
    if hyperparams['model'] == "PPRGo":
        return PPRGoWrapper(**hyperparams)
    return RGNN(**hyperparams)


__all__ = [GCN,
           DenseGCN,
           RGNN,
           RGCN,
           PPRGoWrapperBase,
           PPRGoWrapper,
           RobustPPRGoWrapper,
           create_model,
           MODEL_TYPE,
           BATCHED_PPR_MODELS]
