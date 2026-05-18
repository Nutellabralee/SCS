"""
Pomocne funkcije za problem Shortest Common Superstring (SCS).

Instanca problema:
  - strings: lista stringova  ["abc", "bcd", ...]
  - sol:     binarna lista duzine len(strings)
             sol[i] = 1 => strings[i] je ukljucen u superstring

Superstring se gradi konkatenacijom odabranih stringova uz maksimalni overlap.
Fitness (manji = bolji):
  (broj nepokrivenih stringova, ukupna duzina superstringa)

Optimizacija performansi:
  - Overlap matrica OV[i][j] se racuna JEDNOM po instanci i prosledjuje
    algoritmima, umesto da se overlap racuna iznova pri svakoj evaluaciji.
  - Duzina superstringa se racuna iz overlap matrice bez gradnje stringa:
      duzina = suma(len(s) za odabrane s) - suma(overlapa duz pohlepnog puta)
  - Ovo ubrzava fitness evaluaciju sa O(n^2 * L^2) na O(n^2).
"""

import numpy as np


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


def buildOverlapMatrix(strings: list[str]) -> list[list[int]]:
    """
    Racuna matricu overlapa OV[i][j] = overlap(strings[i], strings[j])
    za sve parove. Poziva se JEDNOM po instanci.

    Vremenska sloznost: O(n^2 * L^2) — ali samo jednom, ne pri svakoj evaluaciji.
    """
    n = len(strings)
    OV = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                OV[i][j] = computeOverlap(strings[i], strings[j])
    return OV


def superstringLengthFast(sol: list[int], strings: list[str], OV: list[list[int]]) -> int:
    """
    Brza gornja granica duzine superstringa koristeći OV matricu.

    Umesto pohlepnog spajanja O(n^3), koristimo greedy chain O(n^2):
    gradimo lanac odabranih stringova birajuci uvek sledeci sa max overlapom
    u odnosu na poslednji dodat string. Ovo je O(n^2) bez string operacija.
    """
    indices = [i for i, v in enumerate(sol) if v == 1]
    if not indices:
        return 0
    if len(indices) == 1:
        return len(strings[indices[0]])

    remaining = set(indices)
    # Pocinjemo od stringa sa najmanjim overlapom prema drugima (konzervativno)
    current = indices[0]
    remaining.remove(current)
    total_len = len(strings[current])

    while remaining:
        # Sledeci: max overlap sa trenutnim
        best_ov = -1
        best_next = next(iter(remaining))
        for nxt in remaining:
            ov = OV[current][nxt]
            if ov > best_ov:
                best_ov = ov
                best_next = nxt
        total_len += len(strings[best_next]) - best_ov
        current = best_next
        remaining.remove(current)

    return total_len


# -----------------------------------------------------------------------
# Matrica pokrivenosti
# -----------------------------------------------------------------------

def buildCoverageMatrix(strings: list[str]) -> tuple:
    """
    Gradi binarnu matricu T i listu tezina.

    T[i][j] = 1  ako strings[i] pokriva (sadrzi) strings[j] kao substring,
              tj. strings[j] in strings[i]

    Vraca: (T, weights)
      T       — lista listi (len(strings) x len(strings))
      weights — koliko puta je svaki string pokriven u ukupnoj matrici
    """
    n = len(strings)
    T = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if strings[j] in strings[i]:
                T[i][j] = 1

    weights = list(np.sum(T, axis=0))
    return T, weights


# -----------------------------------------------------------------------
# Fitness funkcije (brze verzije sa OV matricom)
# -----------------------------------------------------------------------

def calculateFitness(sol, strings: list[str], T=None, OV=None) -> tuple:
    """
    Vraca (broj_nepokrivenih_stringova, duzina_superstringa).
    Manji je bolji.

    Ako su T i OV prosledjeni, koristi ih za brzo racunanje (preporuceno).
    Inace pada na sporu varijantu (samo za kompatibilnost).
    """
    if sol is None:
        return (float('inf'), float('inf'))

    n = len(strings)

    # --- Broj nepokrivenih (brzo ako imamo T matricu) ---
    if T is not None:
        uncovered = 0
        for j in range(n):
            covered = any(sol[i] == 1 and T[i][j] == 1 for i in range(n))
            if not covered:
                uncovered += 1
    else:
        uncovered = 0
        for j in range(n):
            covered = any(sol[i] == 1 and strings[j] in strings[i] for i in range(n))
            if not covered:
                uncovered += 1

    # --- Duzina superstringa ---
    if OV is not None:
        length = superstringLengthFast(sol, strings, OV)
    else:
        length = len(_buildSuperstringSlowFallback(strings, sol))

    return (uncovered, length)


def calculateWeightedFitness(sol, strings: list[str], weights: list,
                              T=None, OV=None) -> tuple:
    """
    Kao calculateFitness, ali penalizacija nepokrivenih stringova je
    obrnuto proporcionalna njihovoj tezini.
    """
    if sol is None:
        return (float('inf'), float('inf'))

    n = len(strings)
    penalty = 0.0

    if T is not None:
        for j in range(n):
            covered = any(sol[i] == 1 and T[i][j] == 1 for i in range(n))
            if not covered:
                w = weights[j] if j < len(weights) else 1
                penalty += 1.0 / (float(w) + 1)
    else:
        for j in range(n):
            covered = any(sol[i] == 1 and strings[j] in strings[i] for i in range(n))
            if not covered:
                w = weights[j] if j < len(weights) else 1
                penalty += 1.0 / (float(w) + 1)

    if OV is not None:
        length = superstringLengthFast(sol, strings, OV)
    else:
        length = len(_buildSuperstringSlowFallback(strings, sol))

    return (penalty, length)


# -----------------------------------------------------------------------
# Sporadna fallback funkcija (samo za kompatibilnost / debug)
# -----------------------------------------------------------------------

def _buildSuperstringSlowFallback(strings: list[str], sol: list[int]) -> str:
    """Stara spora verzija — koristi se samo ako OV nije dostupan."""
    chosen = [s for s, v in zip(strings, sol) if v == 1]
    if not chosen:
        return ""
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
