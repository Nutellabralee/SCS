"""
Genetski algoritam (GA) za SCS problem.

Kodiranje:  binarna lista duzine n — X[i] = 1 ako je strings[i] odabran
Selekcija:  turnirska selekcija
Ukrstanje:  uniformno ukrstanje (uniform crossover)
Mutacija:   bit-flip sa verovatnocom mutationProb
Elitizam:   top elitismRate * populationSize jedinki prenosi se direktno
"""

import numpy as np
from copy import deepcopy
import random

from utils.helpers import calculateFitness


class Individual:
    def __init__(self, strings: list[str], T, code=None):
        self.strings = strings
        self.T = T
        n = len(strings)
        if code is not None:
            self.code = np.array(code, dtype=int)
        else:
            self.code = np.random.randint(0, 2, size=n)
        self.fitness = self._calcFit()

    def _calcFit(self):
        return calculateFitness(list(self.code), self.strings)

    def __lt__(self, other):
        return self.fitness < other.fitness


class GeneticAlgorithm:
    def __init__(
        self,
        populationSize: int,
        numOfGenerations: int,
        mutationProb: float,
        elitismRate: float,
        tournamentSize: int
    ):
        self.populationSize = populationSize
        self.numOfGenerations = numOfGenerations
        self.mutationProb = mutationProb
        self.numOfElites = int(populationSize * elitismRate)
        self.tournamentSize = tournamentSize

        assert 0 <= elitismRate < 1
        # Osiguravamo parnost ostatka populacije
        if self.numOfElites % 2 != self.populationSize % 2:
            self.numOfElites += 1

    def _selection(self, population: list) -> Individual:
        """Turnirska selekcija."""
        participants = random.sample(population, self.tournamentSize)
        return min(participants)

    def _crossover(self, code1: np.ndarray, code2: np.ndarray) -> tuple:
        """Uniformno ukrstanje — svaki gen nezavisno bira roditelja."""
        child1 = np.zeros_like(code1)
        child2 = np.zeros_like(code2)
        for i in range(len(code1)):
            child1[i] = code1[i] if random.random() < 0.5 else code2[i]
            child2[i] = code1[i] if random.random() < 0.5 else code2[i]
        return child1, child2

    def _mutation(self, code: np.ndarray) -> np.ndarray:
        """Bit-flip mutacija."""
        new_code = code.copy()
        for i in range(len(new_code)):
            if random.random() < self.mutationProb:
                new_code[i] = 1 - new_code[i]
        return new_code

    def solve(self, strings: list[str], T) -> tuple:
        """
        Pokrece GA i vraca (solution, fitness, fitnessList).

        fitnessList — fitness najboljeg jedinke kroz generacije
        """
        population = [Individual(strings, T) for _ in range(self.populationSize)]
        fitnessList = [min(population).fitness]

        for _ in range(self.numOfGenerations):
            population.sort()
            newPopulation = population[:self.numOfElites]

            for _ in range(self.numOfElites, self.populationSize, 2):
                parent1 = self._selection(population)
                parent2 = self._selection(population)

                c1_code, c2_code = self._crossover(parent1.code, parent2.code)
                c1_code = self._mutation(c1_code)
                c2_code = self._mutation(c2_code)

                child1 = Individual(strings, T, code=c1_code)
                child2 = Individual(strings, T, code=c2_code)

                newPopulation.append(child1)
                newPopulation.append(child2)

            population = deepcopy(newPopulation)
            fitnessList.append(min(population).fitness)

        best = min(population)
        fitnessList.append(best.fitness)
        return list(best.code), best.fitness, fitnessList
