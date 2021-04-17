from typing import Union
import torch

from .dice import DICE
from .fgsm import FGSM
from .gang import GANG
from .greedy_rbcd import GreedyRBCD
from .local_prbcd import LocalPRBCD
from .local_prbcd_batched import LocalBatchedPRBCD
from .pgd import PGD
from .prbcd import PRBCD
from .nettack import Nettack
from .base_attack import Attack

ATTACK_TYPE = Union[DICE, FGSM, GANG, GreedyRBCD, LocalPRBCD, PGD, PRBCD, Nettack, LocalBatchedPRBCD]
SPARSE_ATTACKS = [GANG.__name__, GreedyRBCD.__name__, PRBCD.__name__, DICE.__name__]
LOCAL_ATTACKS = [LocalPRBCD.__name__, Nettack.__name__, LocalBatchedPRBCD.__name__]


def create_attack(attack: str, binary_attr: bool, attr: torch.Tensor, **kwargs) -> Attack:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    attack : str
        Identifier of the attack
    binary_attr : str
        If true the attributes are binary
    attr : str
        For attr dependent configuration
    kwargs
        Containing the hyperparameters

    Returns
    -------
    Union[FGSM, GANG, GreedyRBCD, PRBCD]
        The created instance
    """
    if not any([attack.lower() == attack_model.__name__.lower() for attack_model in ATTACK_TYPE.__args__]):
        raise ValueError(f'The attack {attack} is not in {ATTACK_TYPE.__args__}')

    kwargs = dict(kwargs)
    if kwargs is None:
        kwargs = {}
    kwargs['X'] = attr

    if binary_attr:
        if 'feature_mode' not in kwargs:
            kwargs['feature_mode'] = 'binary'
    else:
        if 'feature_mode' not in kwargs:
            kwargs['feature_mode'] = 'symmetric_float'
        if 'feature_max_abs' not in kwargs:
            kwargs['feature_max_abs'] = attr.abs().max().item()

    return globals()[attack](**kwargs)


__all__ = [FGSM, GANG, GreedyRBCD, LocalPRBCD, LocalBatchedPRBCD,
           PRBCD, create_attack, ATTACK_TYPE, SPARSE_ATTACKS, Nettack]
