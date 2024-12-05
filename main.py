from orthogonal_matrix import matrices1_ledm, matrices2_slackngon
from grasp import metaheuristic
from datetime import datetime
from numpy import ndarray, loadtxt


# Matrice cible
M: ndarray

M = matrices1_ledm(3)
M = matrices2_slackngon(3)
M = loadtxt('matrice_examples/matrice30.txt', dtype=int)


# Paramètres de la métaheuristique
grasp_max_iterations = 10
grasp_alpha = 0.1

vns_k_max = 3


# Temps de début
starttime: datetime = datetime.now()
print("\n||===================== GRASP =====================||")
print(f"|| Grasp started on {starttime.date()} at {starttime.time()}  ||")
print("||=================================================||\n||")

# Metaheuristique
best_pattern, best_fobj = metaheuristic(M, grasp_max_iterations, grasp_alpha, vns_k_max)
print(f"|| Best pattern found: ")
for row in best_pattern:
    print(f"|| {row}")

print(f"||\n|| Best Matrix Rank: {best_fobj[0]}")
print(f"|| Lowest Single Value: {best_fobj[1]}")


# Temps de fin
endtime: datetime = datetime.now()
duration: datetime = endtime - starttime
print(f"||\n|| Duration: {duration.seconds // 60}m {duration.seconds % 60}s {duration.microseconds}µs")

print("||\n||=================================================||")
print(f"|| Grasp ended on {endtime.date()} at {endtime.time()}    ||")
print("||=================================================||\n\n")
