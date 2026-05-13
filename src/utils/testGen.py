"""
Generisanje i ucitavanje test instanci za problem Shortest Common Superstring.

Format JSON fajla:
{
  "n": <broj stringova>,
  "strings": ["abc", "bcd", ...]
}

Instanca je validna ako postoji feasibilno resenje (uvek postoji —
konkatenacija svih stringova je trivijalni superstring).
"""

import random
import json
import os
import string

from algorithms.greedy import greedy
from utils.helpers import buildCoverageMatrix


# -----------------------------------------------------------------------
# Generisanje instanci
# -----------------------------------------------------------------------

ALPHABET = string.ascii_lowercase


def generateRandomInstance(
    numOfStrings: int,
    minLen: int,
    maxLen: int,
    alphabetSize: int = 4,
    seed=None
) -> list[str]:
    """
    Generise listu `numOfStrings` nasumicnih stringova
    duzine u opsegu [minLen, maxLen] nad azbukom velicine `alphabetSize`.
    """
    if seed is not None:
        random.seed(seed)

    alpha = ALPHABET[:alphabetSize]
    strings = []
    for _ in range(numOfStrings):
        length = random.randint(minLen, maxLen)
        s = ''.join(random.choices(alpha, k=length))
        strings.append(s)
    return strings


def saveInstance(filename: str, strings: list[str]):
    data = {
        "n": len(strings),
        "strings": strings
    }
    with open(filename, "w") as f:
        json.dump(data, f)


def loadInstance(filename: str) -> list[str]:
    with open(filename, "r") as f:
        data = json.load(f)
    return data["strings"]


# -----------------------------------------------------------------------
# Kreiranje skupa testova
# -----------------------------------------------------------------------

def makeTests(
    directoryPath: str,
    numOfTests: int,
    numOfStringsRange: list,
    minLenRange: list,
    maxLenRange: list,
    alphabetSize: int = 4,
    seed=None
):
    """
    Generise `numOfTests` validnih instanci i snima ih u `directoryPath`.
    Instanca je uvek validna (SCS uvek postoji), ali proveravamo
    da greedy daje feasibilno resenje (duzina > 0).
    """
    i = 1
    while i <= numOfTests:
        numOfStrings = random.choice(numOfStringsRange)
        minLen = random.choice(minLenRange)
        maxLen = random.choice(maxLenRange)
        if maxLen < minLen:
            minLen, maxLen = maxLen, minLen

        strings = generateRandomInstance(numOfStrings, minLen, maxLen, alphabetSize, seed)

        # Provera: greedy mora dati resenje
        T, weights = buildCoverageMatrix(strings)
        sol, fitness = greedy(strings, T)

        if fitness[0] == 0:  # svi stringovi pokriveni
            filepath = os.path.join(os.getcwd(), directoryPath, f"{i}.json")
            print(f"Test {i}: n={numOfStrings}, lenRange=[{minLen},{maxLen}]")
            saveInstance(filepath, strings)
            i += 1
