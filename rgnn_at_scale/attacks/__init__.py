from typing import Union

from .fgsm import FGSM
from .gang import GANG
from .greedy_rbcd import GreedyRBCD
from .prbcd import PRBCD
ATTACK_TYPE = Union[FGSM, GANG, GreedyRBCD, PRBCD]
SPARSE_ATTACKS = [GANG.__name__, GreedyRBCD.__name__, PRBCD.__name__]


def create_attack(attack: str, **kwargs) -> ATTACK_TYPE:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    attack : str
        Identifier of the attack
    kwargs
        Containing the hyperparameters

    Returns
    -------
    Union[FGSM, GANG, GreedyRBCD, PRBCD]
        The created instance
    """
    if not any([attack == attack_model.__name__ for attack_model in ATTACK_TYPE.__args__]):
        raise ValueError(f'The attack {attack} is not in {ATTACK_TYPE.__args__}')
    return globals()[attack](**kwargs)


__all__ = [FGSM, GANG, GreedyRBCD, PRBCD, create_attack, ATTACK_TYPE, SPARSE_ATTACKS]
