from typing import Any, Dict, Union

from rgnn_at_scale.models.gcn import GCN
from rgnn_at_scale.models.rgnn import RGNN
from rgnn_at_scale.models.rgcn import RGCN
from rgnn_at_scale.models.pprgo import RobustPPRGoWrapper, PPRGoWrapper, PPRGoDiffEmbWrapper, RobustPPRGoDiffEmbWrapper


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
    if hyperparams['model'] == 'RGCN':
        return RGCN(**hyperparams)
    elif hyperparams['model'] == "PPRGoDiffEmbWrapper":
        return PPRGoDiffEmbWrapper(**hyperparams)
    elif hyperparams['model'] == "RobustPPRGoDiffEmb":
        return RobustPPRGoDiffEmbWrapper(**hyperparams)
    elif hyperparams['model'] == "RobustPPRGo":
        return RobustPPRGoWrapper(**hyperparams)
    elif hyperparams['model'] == "PPRGo":
        return PPRGoWrapper(**hyperparams)
    else:
        return RGNN(**hyperparams)


__all__ = [GCN,
           RGNN,
           RGCN,
           PPRGoWrapper,
           RobustPPRGoWrapper,
           PPRGoDiffEmbWrapper,
           RobustPPRGoDiffEmbWrapper,
           create_model,
           MODEL_TYPE,
           BATCHED_PPR_MODELS]
