from orthogonal_matrix import matrices1_ledm, matrices2_slackngon
from grasp import metaheuristic_grasp
from datetime import datetime, timedelta
from numpy import ndarray, random, loadtxt


# Definition d'une Seed pour la reproductibilité 
# random.seed(42)


# Matrice cible
M: ndarray

# M = matrices1_ledm(5)
# M = matrices2_slackngon(7)
M = loadtxt('matrice_examples/correl5_matrix.txt', dtype=int)


# Paramètres de la métaheuristique
grasp_max_iterations = 5000
grasp_alpha = 0.001

vns_k_max = 2


# Temps de début
starttime: datetime = datetime.now()
print("\n\n||================================ GRASP ================================||")
print(f"|| Grasp started on {starttime.date()} at {starttime.time()}                        ||")
print("||=======================================================================||")


# Metaheuristique
best_pattern, best_fobj = metaheuristic_grasp(M, grasp_max_iterations, grasp_alpha, vns_k_max)
print(f"||\n|| Best pattern found: ")
for row in best_pattern:
    print(f"|| {row}")

print(f"||\n|| Best Matrix Rank: {best_fobj[0]}")
print(f"|| Lowest Single Value: {best_fobj[1]}")


# Temps de fin
endtime: datetime = datetime.now()
duration: timedelta = endtime - starttime
print(f"||\n|| Duration: {duration.seconds // 60}m {duration.seconds % 60}s {duration.microseconds}µs")

print("||\n||================================ GRASP ================================||")
print(f"|| Grasp ended on {endtime.date()} at {endtime.time()}                          ||")
print("||=======================================================================||\n\n")
