from typing import Any, Dict, Union

from rgnn_at_scale.models.gat import RGAT
from rgnn_at_scale.models.gcn import GCN, DenseGCN
from rgnn_at_scale.models.rgnn import RGNN
from rgnn_at_scale.models.rgcn import RGCN
from rgnn_at_scale.models.pprgo import (PPRGoWrapperBase, RobustPPRGoWrapper,
                                        PPRGoWrapper, PPRGoDiffEmbWrapper, RobustPPRGoDiffEmbWrapper)


MODEL_TYPE = Union[GCN, RGNN, RGCN, RobustPPRGoWrapper, PPRGoWrapper]
BATCHED_PPR_MODELS = Union[RobustPPRGoWrapper, PPRGoWrapper, PPRGoDiffEmbWrapper, RobustPPRGoDiffEmbWrapper]


def create_model(hyperparams: Dict[str, Any]) -> MODEL_TYPE:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    hyperparams : Dict[str, Any]
        Containing the hyperparameters.

    Returns
    -------
    Union[GCN, RGNN]
        The created instance.
    """
    if 'model' not in hyperparams or hyperparams['model'] == 'GCN':
        return GCN(**hyperparams)
    if hyperparams['model'] == 'RGAT':
        return RGAT(**hyperparams)
    if hyperparams['model'] == 'DenseGCN':
        return DenseGCN(**hyperparams)
    if hyperparams['model'] == 'RGCN':
        return RGCN(**hyperparams)
    if hyperparams['model'] == "PPRGoDiffEmbWrapper":
        return PPRGoDiffEmbWrapper(**hyperparams)
    if hyperparams['model'] == "RobustPPRGoDiffEmb":
        return RobustPPRGoDiffEmbWrapper(**hyperparams)
    if hyperparams['model'] == "RobustPPRGo":
        return RobustPPRGoWrapper(**hyperparams)
    if hyperparams['model'] == "PPRGo":
        return PPRGoWrapper(**hyperparams)
    return RGNN(**hyperparams)


__all__ = [GCN,
           DenseGCN,
           RGAT,
           RGNN,
           RGCN,
           PPRGoWrapperBase,
           PPRGoWrapper,
           RobustPPRGoWrapper,
           PPRGoDiffEmbWrapper,
           RobustPPRGoDiffEmbWrapper,
           create_model,
           MODEL_TYPE,
           BATCHED_PPR_MODELS]
