"""
Pohlepni (Greedy) algoritam za SCS problem.

Strategija:
  U svakom koraku biramo string koji pokriva najveci broj
  jos nepokrivenih stringova. Ponavljamo dok svi stringovi
  nisu pokriveni ili vise nema napretka.
"""

from utils.helpers import calculateFitness


def greedy(strings: list[str], T) -> tuple:
    """
    Pohlepno resava SCS problem.

    Parametri
    ---------
    strings : lista stringova
    T       : matrica pokrivenosti — T[i][j] = 1 ako strings[i] pokriva strings[j]

    Vraca
    -----
    (solution, fitness)
      solution — binarna lista duzine n
      fitness  — (uncovered, superstring_length)
    """
    n = len(strings)
    solution = [0] * n
    uncovered = set(range(n))   # indeksi jos nepokrivenih stringova

    # coverage[i] = skup indeksa koje strings[i] pokriva
    coverage = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if T[i][j] == 1:
                coverage[i].add(j)

    while uncovered:
        # Biramo string koji pokriva najvise jos nepokrivenih
        bestIdx = max(
            range(n),
            key=lambda i: len(coverage[i] & uncovered) if solution[i] == 0 else -1
        )

        # Ako ni jedan ne pokriva nista vise, stajemo
        if len(coverage[bestIdx] & uncovered) == 0:
            break

        solution[bestIdx] = 1
        uncovered -= coverage[bestIdx]

    return solution, calculateFitness(solution, strings)
