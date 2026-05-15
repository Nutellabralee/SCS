"""
Variable Neighbourhood Search (VNS) algoritam za SCS problem.

Struktura:
  shaking    — k random bit-swap-ova (perturbacija)
  localSearch — bit-flip lokalna pretraga (first improvement)

Podrska za obicni i weighted fitness.
"""

from copy import deepcopy
import random
from typing import Optional

from utils.helpers import calculateFitness, calculateWeightedFitness


def localSearch(
    solution: list,
    fitness,
    strings: list[str],
    weighted: bool = False,
    weights: Optional[list] = None,
    T=None,
    OV=None,
):
    """
    First-improvement lokalna pretraga sa bit-flip operatorom.
    """
    improved = True
    while improved:
        improved = False
        indices = list(range(len(solution)))
        random.shuffle(indices)
        for i in indices:
            solution[i] = 1 - solution[i]
            if weighted and weights is not None:
                newFitness = calculateWeightedFitness(solution, strings, weights, T, OV)
            else:
                newFitness = calculateFitness(solution, strings, T, OV)

            if newFitness < fitness:
                fitness = newFitness
                improved = True
                break
            else:
                solution[i] = 1 - solution[i]

    return fitness


def shaking(solution: list, k: int) -> list:
    """k nasumicnih swap-ova (perturbacija)."""
    assert len(solution) >= max(2, k)
    new = deepcopy(solution)
    for _ in range(k):
        i, j = random.sample(range(len(solution)), 2)
        new[i], new[j] = new[j], new[i]
    return new


def vns(
    strings: list[str],
    T,
    numOfIters: int,
    minK: int,
    maxK: int,
    moveProb: float,
    weighted: bool = False,
    weights: Optional[list] = None,
    OV=None,
) -> tuple:
    """
    Pokrece VNS i vraca (solution, fitness, fitnessList).
    """
    n = len(strings)
    solution = random.choices([0, 1], k=n)

    if weighted and weights is not None:
        fitness = calculateWeightedFitness(solution, strings, weights, T, OV)
    else:
        fitness = calculateFitness(solution, strings, T, OV)

    fitnessList = [fitness]

    for _ in range(numOfIters):
        for k in range(minK, min(maxK + 1, n)):
            newSolution = shaking(solution, k)

            if weighted and weights is not None:
                newFitness = calculateWeightedFitness(newSolution, strings, weights, T, OV)
            else:
                newFitness = calculateFitness(newSolution, strings, T, OV)

            newFitness = localSearch(newSolution, newFitness, strings, weighted, weights, T, OV)

            if newFitness < fitness or (newFitness == fitness and random.random() < moveProb):
                fitness = newFitness
                solution = deepcopy(newSolution)
                break

        fitnessList.append(fitness)

    return solution, fitness, fitnessList
