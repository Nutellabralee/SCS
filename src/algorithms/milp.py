"""
MILP (Mixed Integer Linear Programming) algoritam za SCS problem.

Koristi IBM CPLEX (docplex) za egzaktno resavanje.
Model:
  Promenljive: X[i] in {0,1}  — da li je strings[i] odabran
  Ogranicenja: za svaki string j, bar jedan odabrani string i ga mora pokrivati:
               sum(X[i] * T[i][j] for i) >= 1  za svako j
  Cilj:        minimizovati ukupnu duzinu superstringa
               (aproksimiramo je zbirom duzina odabranih stringova kao gornja granica,
                jer egzaktan model zahteva nelinearne promenljive za overlap)

Napomena: docplex zahteva IBM CPLEX Optimization Studio (studentska licenca).
"""

try:
    from docplex.mp.model import Model
    from docplex.mp.utils import DOcplexException
    CPLEX_AVAILABLE = True
except ImportError:
    CPLEX_AVAILABLE = False


def milp(strings: list[str], T, weights) -> tuple:
    """
    Resava SCS problem pomocu MILP-a.

    Ako CPLEX nije dostupan, vraca None (graceful fallback).

    Vraca
    -----
    (solution, value, status)
      solution — binarna lista ili None
      value    — ukupna duzina superstringa ili inf
      status   — string opis statusa resavaca
    """
    if not CPLEX_AVAILABLE:
        return None, float('inf'), 'cplex_not_available'

    n = len(strings)
    model = Model("SCS")

    X = [model.binary_var(name=f"x{i}") for i in range(n)]

    # Svaki string mora biti pokriven bar jednim odabranim stringom
    for j in range(n):
        try:
            model.add_constraint(
                model.sum(X[i] * T[i][j] for i in range(n)) >= 1,
                ctname=f"cover_{j}"
            )
        except DOcplexException:
            return None, float('inf'), 'trivially_infeasible'

    # Ciljna funkcija: minimizovati zbir duzina odabranih stringova
    # (gornja granica duzine superstringa)
    model.minimize(
        model.sum(X[i] * len(strings[i]) for i in range(n))
    )

    solution = model.solve(log_output=False)
    if solution:
        sol = [int(x.solution_value) for x in X]
        return sol, int(solution.get_objective_value()), model.solve_details.status

    return None, float('inf'), model.solve_details.status
