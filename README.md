# Shortest Common Superstring


## Opis projekta

Projekat predstavlja implementaciju i eksperimentalno poređenje više algoritama za rešavanje problema **Shortest Common Superstring (SCS)**. Projekat je razvijen u okviru kursa Računarska inteligencija na Matematičkom fakultetu.

Dat je skup stringova `S = {s₁, s₂, ..., sₙ}`. **Superstring** je svaki string `T` takav da se svaki `sᵢ` pojavljuje kao podstring od `T`. Cilj je pronaći najkraći takav string.

SCS je **NP-težak problem** (Gallant et al., 1980) sa primenama u sekvenciranju genoma, kompresiji podataka i VLSI testiranju.


## Instalacije

Projekat je implementiran u programskom jeziku Python. Za pokretanje koda potrebno je instalirati sledeće biblioteke:

```
pip install numpy pandas matplotlib jupyter
```

Za MILP algoritam potreban je **IBM CPLEX Optimization Studio** (studentska licenca) i `docplex` paket. Ostali algoritmi rade bez CPLEX-a.


## Korišćenje

Projekat sadrži implementacije algoritama i biblioteku za generisanje test instanci. Eksperimentalni deo je automatizovan kroz Jupyter notebook.

Za pokretanje koda:
1. Otvoriti `src/Main.ipynb`
2. Podesiti broj test instanci koje se generišu u pozivima funkcije `makeTests`
3. Izvršiti notebook ćeliju po ćeliju


## Implementirani algoritmi

| Algoritam | Tip | Složenost | Skalabilnost |
|-----------|-----|-----------|--------------|
| Brute Force | Egzaktni | O(2ⁿ) | malo (n ≤ 20) |
| MILP (CPLEX) | Egzaktni | - | malo/srednje |
| Greedy | Aproksimativni | O(n²) | sve veličine |
| Genetic Algorithm | Metaheuristika | O(gen·pop·n) | sve veličine |
| VNS | Metaheuristika | O(iter·n²) | sve veličine |
| VNS Weighted | Metaheuristika | O(iter·n²) | sve veličine |


## Struktura projekta

```
shortest-common-superstring/
├── data/
│   ├── small/          # 30 malih instanci (5–12 stringova)
│   ├── medium/         # 15 srednjih instanci (15–30 stringova)
│   └── large/          # 5 velikih instanci (50–100 stringova)
├── results/
│   ├── small/          # Rezultati po instanci
│   ├── medium/
│   ├── large/
│   ├── smallMetrics.json
│   ├── mediumMetrics.json
│   └── largeMetrics.json
├── src/
│   ├── Main.ipynb      # Jupyter notebook — glavni ulaz
│   ├── algorithms/
│   │   ├── bruteForce.py
│   │   ├── greedy.py
│   │   ├── milp.py
│   │   ├── genetic.py
│   │   └── vns.py
│   └── utils/
│       ├── helpers.py   # Fitness funkcije, overlap, matrica pokrivenosti
│       ├── testGen.py   # Generisanje i učitavanje instanci
│       └── testLib.py   # Pokretanje eksperimenata i metrike
└── README.md
```


## Dokumentacija

Rezultati i prateća dokumentacija nalaze se u `docs/`.
