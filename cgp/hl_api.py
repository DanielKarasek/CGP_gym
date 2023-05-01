import warnings
from typing import Callable, Optional

import numpy as np

from .ea import MuPlusLambda
from .individual import IndividualBase
from .population import Population

#xkaras38 Function was here, but I split it and got rid of stuff 
# I didn't care for
def evolve(
    pop: Population,
    objective: Callable[[IndividualBase], IndividualBase],
    ea: MuPlusLambda
) -> None:
    """Evolve init function, inits pop and does 1 evo step
    ______________________________________________________

    :param pop: cgp.Population
    :param objective: Callable
    :param ea: ea.MuPlusLambda
    -------------------------
    :return:
    """
    ea.initialize_fitness_parents(pop, objective)
    for ind in pop.parents:
        ind.orig_fitness = ind.fitness

    pop = ea.step(pop, objective)


#xkaras38
def evolve_continue(
    pop: Population,
    objective: Callable[[IndividualBase], IndividualBase],
    ea: MuPlusLambda,
    terminal_fitness: Optional[float] = None
) -> bool:
    """
    Little bit of bypass for evolution, so I can evolve and
    simultaneously change parameters.
    _______________________________________________________


    :param pop: cgp.Population
    :param objective: Callable
    :param ea: ea.MuPlusLambda
    -------------------------
    :return:
    """
    pop = ea.step(pop, objective)
    if terminal_fitness is not None:
        if pop.champion.orig_fitness >= terminal_fitness:
            return True
    return False
