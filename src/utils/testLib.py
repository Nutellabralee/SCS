"""
Biblioteka za pokretanje eksperimenata nad instancama SCS problema.

Analogno strukturi originalnog projekta:
  test()        — testira jednu instancu svim algoritmima
  testAll()     — testira sve instance u folderu
  saveResult()  — snima rezultate jedne instance
  calculateMetrics() — agregira metrike
  saveMetrics() — snima metrike
"""

import time
import os
import json
import numpy as np

from utils.helpers import calculateFitness, buildCoverageMatrix
from utils.testGen import loadInstance

from algorithms.bruteForce import bruteForce
from algorithms.greedy import greedy
from algorithms.milp import milp
from algorithms.genetic import GeneticAlgorithm
from algorithms.vns import vns


def test(strings: list[str], scale: str):
    """
    Testira jednu instancu svim algoritmima.

    Vraca:
      result          — dict: algo -> (sol, value, time, isFeasible)
      fitnessListGA   — lista fitness vrednosti po generacijama (GA)
      fitnessListVns  — lista fitness vrednosti po iteracijama (VNS)
      fitnessListVnsW — lista fitness vrednosti po iteracijama (VNS weighted)
    """
    T, weights = buildCoverageMatrix(strings)
    result = {}

    # ---- Brute Force (samo za male instance) ----
    if scale == "small":
        startTime = time.perf_counter()
        sol, fitness = bruteForce(strings, T)
        endTime = time.perf_counter()
        result["bruteForce"] = (sol, fitness[1], endTime - startTime, fitness[0] == 0)

    # ---- MILP (male i srednje instance) ----
    if scale in ("small", "medium"):
        startTime = time.perf_counter()
        sol, value, details = milp(strings, T, weights)
        endTime = time.perf_counter()
        isFeasible = False
        if sol is not None:
            fitness = calculateFitness(sol, strings)
            isFeasible = fitness[0] == 0
        result["milp"] = (sol, value, endTime - startTime, isFeasible)

    # ---- Greedy ----
    startTime = time.perf_counter()
    sol, fitness = greedy(strings, T)
    endTime = time.perf_counter()
    result["greedy"] = (sol, fitness[1], endTime - startTime, fitness[0] == 0)

    # ---- Genetic Algorithm ----
    if scale == "small":
        ga = GeneticAlgorithm(30, 40, 0.05, 0.2, 4)
    elif scale == "medium":
        ga = GeneticAlgorithm(80, 100, 0.05, 0.1, 8)
    else:
        ga = GeneticAlgorithm(100, 140, 0.05, 0.1, 10)

    startTime = time.perf_counter()
    sol, fitness, fitnessListGA = ga.solve(strings, T)
    endTime = time.perf_counter()
    result["genetic"] = (sol, fitness[1], endTime - startTime, fitness[0] == 0)

    # ---- VNS ----
    startTime = time.perf_counter()
    if scale == "small":
        sol, fitness, fitnessListVns = vns(strings, T, 30, 1, 3, 0.05)
    elif scale == "medium":
        sol, fitness, fitnessListVns = vns(strings, T, 50, 1, 8, 0.1)
    else:
        sol, fitness, fitnessListVns = vns(strings, T, 80, 1, 12, 0.2)
    endTime = time.perf_counter()
    result["vns"] = (sol, fitness[1], endTime - startTime, fitness[0] == 0)

    # ---- VNS Weighted ----
    startTime = time.perf_counter()
    if scale == "small":
        sol, fitness, fitnessListVnsW = vns(strings, T, 30, 1, 3, 0.05, True, weights)
    elif scale == "medium":
        sol, fitness, fitnessListVnsW = vns(strings, T, 50, 1, 8, 0.1, True, weights)
    else:
        sol, fitness, fitnessListVnsW = vns(strings, T, 80, 1, 12, 0.2, True, weights)
    endTime = time.perf_counter()
    result["vnsWeighted"] = (sol, fitness[1], endTime - startTime, fitness[0] == 0)

    return result, fitnessListGA, fitnessListVns, fitnessListVnsW


def saveResult(result, filepath: str):
    serialized = {}
    for algo, data in result.items():
        sol, value, t, isFeasible = data
        serialized[algo] = {
            "solution": [int(x) for x in sol] if sol is not None else None,
            "value": int(value) if value != float('inf') else value,
            "time": t,
            "isFeasible": bool(isFeasible)
        }
    with open(filepath, "w") as f:
        json.dump(serialized, f, indent=2)


def calculateMetrics(results: list) -> dict:
    sums = {}
    for result in results:
        compareToExact = False
        exactValue = None
        if 'milp' in result:
            _, value, _, isFeasible = result['milp']
            if isFeasible:
                compareToExact = True
                exactValue = value

        for algo, (sol, value, t, isFeasible) in result.items():
            if algo not in sums:
                sums[algo] = {"values": [], "times": [], "feasible": [], "optimal": []}

            if isFeasible:
                sums[algo]["values"].append(value)
            sums[algo]["times"].append(t)
            sums[algo]["feasible"].append(isFeasible)
            if compareToExact:
                if isFeasible:
                    sums[algo]["optimal"].append(value <= exactValue)
                else:
                    sums[algo]["optimal"].append(False)

    metrics = {}
    for algo, data in sums.items():
        metrics[algo] = {
            "averageFitness": float(np.mean(data["values"])) if data["values"] else None,
            "averageTime": float(np.mean(data["times"])),
            "percentOfFeasible": 100 * float(np.mean(data["feasible"])),
            "percentOfOptimal": 100 * float(np.mean(data["optimal"])) if data["optimal"] else None
        }
    return metrics


def saveMetrics(metrics: dict, metricsPath: str):
    with open(metricsPath, "w") as f:
        json.dump(metrics, f, indent=2)


def testAll(scale: str):
    dataDirPath = os.path.join("../data", scale)
    resultsDirPath = os.path.join("../results", scale)
    results = []

    for fileName in sorted(os.listdir(dataDirPath)):
        dataPath = os.path.join(dataDirPath, fileName)
        if not os.path.isfile(dataPath):
            continue

        strings = loadInstance(dataPath)
        result, _, _, _ = test(strings, scale)
        results.append(result)

        resultPath = os.path.join(resultsDirPath, fileName)
        saveResult(result, resultPath)

    metrics = calculateMetrics(results)
    metricsPath = os.path.join("../results", scale + "Metrics.json")
    saveMetrics(metrics, metricsPath)

    return metrics
