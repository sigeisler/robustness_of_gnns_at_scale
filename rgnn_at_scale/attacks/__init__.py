from typing import Union

from .dice import DICE
from .fgsm import FGSM
from .gang import GANG
from .greedy_rbcd import GreedyRBCD
from .pgd import PGD
from .prbcd import PRBCD
from .contract_attack import EXPAND_CONTRACT
ATTACK_TYPE = Union[DICE, FGSM, GANG, GreedyRBCD, PGD, PRBCD, EXPAND_CONTRACT]
SPARSE_ATTACKS = [GANG.__name__, GreedyRBCD.__name__, PRBCD.__name__, DICE.__name__]


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
    '''found = False
    for attack_model in ATTACK_TYPE.__args__ :
        print(f'This attack is {attack} and the current name is {attack_model.__name__}')
        if attack.lower() == attack_model.__name__.lower():
            found = True
            break
    if found:
        return globals()[attack](**kwargs)
    else:
        raise ValueError(f'The attack {attack} is not in {ATTACK_TYPE.__args__}')'''

    if not any([attack.lower() == attack_model.__name__.lower() for attack_model in ATTACK_TYPE.__args__]):
        raise ValueError(f'The attack {attack} is not in {ATTACK_TYPE.__args__}')
    return globals()[attack.upper()](**kwargs)


__all__ = [FGSM, GANG, GreedyRBCD, PRBCD, create_attack, ATTACK_TYPE, SPARSE_ATTACKS]
