"""
Pomocne funkcije za problem Shortest Common Superstring (SCS).

Instanca problema:
  - strings: lista stringova  ["abc", "bcd", ...]
  - sol:     binarna lista duzine len(strings)
             sol[i] = 1 => strings[i] je ukljucen u superstring

Superstring se gradi konkatenacijom odabranih stringova uz maksimalni overlap.
Fitness (manji = bolji):
  (broj nepokrivenih stringova, ukupna duzina superstringa)
"""

from itertools import combinations


# -----------------------------------------------------------------------
# Overlap / merge
# -----------------------------------------------------------------------

def computeOverlap(a: str, b: str) -> int:
    """Vraca duzinu najduzeg sufiksa 'a' koji je prefiks 'b'."""
    limit = min(len(a), len(b))
    for length in range(limit, 0, -1):
        if a.endswith(b[:length]):
            return length
    return 0


def buildSuperstring(strings: list[str], sol: list[int]) -> str:
    """
    Od odabranih stringova (sol[i]==1) gradi superstring
    pohlepnim redosledom maksimalnog overlapa.
    Vraca prazan string ako nijedan nije odabran.
    """
    chosen = [s for s, v in zip(strings, sol) if v == 1]
    if not chosen:
        return ""
    # greedy merge
    pool = list(chosen)
    while len(pool) > 1:
        best_ov = -1
        bi, bj = 0, 1
        for i in range(len(pool)):
            for j in range(len(pool)):
                if i == j:
                    continue
                ov = computeOverlap(pool[i], pool[j])
                if ov > best_ov:
                    best_ov, bi, bj = ov, i, j
        merged = pool[bi] + pool[bj][best_ov:]
        pool = [pool[k] for k in range(len(pool)) if k != bi and k != bj]
        pool.append(merged)
    return pool[0]


# -----------------------------------------------------------------------
# Matrica pokrivenosti
# -----------------------------------------------------------------------

def buildCoverageMatrix(strings: list[str]) -> tuple:
    """
    Gradi binarnu matricu T i listu tezina.

    T[i][j] = 1  ako strings[i] pokriva (sadrzi) strings[j] kao substring,
              tj. strings[j] in strings[i]

    Ovde 'pokrivanje' znaci: odabirom strings[i] (sol[i]=1)
    pokrivamo sve strings[j] koji su mu podstringovi.

    Vraca: (T, weights)
      T       — lista listi (len(strings) x len(strings))
      weights — koliko puta je svaki string pokriven u ukupnoj matrici
    """
    n = len(strings)
    T = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if strings[j] in strings[i]:   # strings[i] pokriva strings[j]
                T[i][j] = 1

    import numpy as np
    weights = list(np.sum(T, axis=0))
    return T, weights


# -----------------------------------------------------------------------
# Fitness funkcije
# -----------------------------------------------------------------------

def calculateFitness(sol, strings: list[str]) -> tuple:
    """
    Vraca (broj_nepokrivenih_stringova, duzina_superstringa).
    Manji je bolji.

    Nepokriveni string = strings[j] za koji ne postoji i sa sol[i]=1
    i strings[j] in strings[i].
    Ako je sol prazan (svi nule), svi stringovi su nepokriveni.
    """
    if sol is None:
        return (float('inf'), float('inf'))

    n = len(strings)
    uncovered = 0
    for j in range(n):
        # strings[j] je pokriven ako ga bar jedan odabrani string sadrzi
        covered = any(sol[i] == 1 and strings[j] in strings[i] for i in range(n))
        if not covered:
            uncovered += 1

    superstr = buildSuperstring(strings, sol)
    return (uncovered, len(superstr))


def calculateWeightedFitness(sol, strings: list[str], weights: list) -> tuple:
    """
    Kao calculateFitness, ali penalizacija nepokrivenih stringova je
    obrnuto proporcionalna njihovoj tezini (cescim stringovima se
    daje manja penalizacija jer ih je lakse pokriti drugima).
    """
    if sol is None:
        return (float('inf'), float('inf'))

    n = len(strings)
    penalty = 0.0
    for j in range(n):
        covered = any(sol[i] == 1 and strings[j] in strings[i] for i in range(n))
        if not covered:
            w = weights[j] if j < len(weights) else 1
            penalty += 1.0 / (w + 1)

    superstr = buildSuperstring(strings, sol)
    return (penalty, len(superstr))
