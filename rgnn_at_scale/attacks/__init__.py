from typing import Union

import torch

from .dice import DICE
from .fgsm import FGSM
from .gang import GANG
from .greedy_rbcd import GreedyRBCD
from .pgd import PGD
from .prbcd import PRBCD
ATTACK_TYPE = Union[DICE, FGSM, GANG, GreedyRBCD, PGD, PRBCD]
SPARSE_ATTACKS = [GANG.__name__, GreedyRBCD.__name__, PRBCD.__name__, DICE.__name__]


def create_attack(attack: str, binary_attr: bool, attr: torch.Tensor, **kwargs) -> ATTACK_TYPE:
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
    if not any([attack == attack_model.__name__ for attack_model in ATTACK_TYPE.__args__]):
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


__all__ = [FGSM, GANG, GreedyRBCD, PRBCD, create_attack, ATTACK_TYPE, SPARSE_ATTACKS]