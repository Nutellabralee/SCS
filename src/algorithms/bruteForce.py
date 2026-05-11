"""
Brute Force algoritam za SCS problem.

Prolazi kroz sve 2^n binarne kombinacije i bira onu
koja daje minimalni fitness (najpre minimalan broj nepokrivenih,
a zatim minimalnu duzinu superstringa).

Prakticno samo za n <= 20.
"""

from itertools import product
from utils.helpers import calculateFitness


def bruteForce(strings: list[str], T) -> tuple:
    """
    Pronalazi optimalno resenje exhaustive pretragom.

    Parametri
    ---------
    strings : lista stringova (instanca problema)
    T       : matrica pokrivenosti (nije direktno potrebna ovde,
              ali se prosledi radi konzistentnosti sa ostalim algoritmima)

    Vraca
    -----
    (bestSolution, bestFitness)
      bestSolution — binarna lista duzine n
      bestFitness  — (uncovered, superstring_length)
    """
    n = len(strings)
    bestSolution = [0] * n
    bestFitness = (float('inf'), float('inf'))

    for sol in product(range(2), repeat=n):
        fitness = calculateFitness(list(sol), strings)
        if fitness < bestFitness:
            bestFitness = fitness
            bestSolution = list(sol)

    return bestSolution, bestFitness
