from datetime import datetime, timedelta
from numpy import ndarray, random, loadtxt, logspace
import matplotlib.pyplot as plt

from utils.orthogonal_matrix import matrices1_ledm, matrices2_slackngon
from optimisation.grasp import metaheuristic_grasp

from optimisation.genetic import genetic_algorithm

from options import GraspOptions, MatrixOptions, GraspConstructionOptions

def plot(iteration_list, rank_list, singular_value_list): 
    # Affichage des graphiques
    plt.figure(figsize=(12, 6))

    # Évolution du rang
    plt.subplot(1, 2, 1)
    plt.plot(iteration_list, rank_list, marker='o', label='Rang')
    plt.xlabel("Itérations")
    plt.ylabel("Rang")
    plt.title("Évolution du Rang")
    plt.grid()
    plt.legend()

    # Évolution de la plus petite valeur singulière
    plt.subplot(1, 2, 2)
    plt.plot(iteration_list, singular_value_list, marker='o', color='orange', label='Valeur singulière')
    plt.xlabel("Itérations")
    plt.ylabel("Plus petite valeur singulière")
    plt.title("Évolution de la plus petite valeur singulière")
    plt.grid()
    plt.legend()

    # Affichage final
    plt.tight_layout()
    plt.show()

# Definition d'une Seed pour la reproductibilité 
random.seed(42)

# Paramètres de l'algorithme
matrix_options = MatrixOptions()
grasp_options = GraspOptions()

# Matrice cible
M: ndarray

M = matrices1_ledm(matrix_options.n)
# M = matrices2_slackngon(matrix_options.n)
# M = loadtxt('matrice_examples/correl5_matrix.txt', dtype=int)


# Temps de début
starttime: datetime = datetime.now()
print("\n\n||================================ GRASP ================================||")
print(f"|| Grasp started on {starttime.date()} at {starttime.time()}                        ||")
print("||=======================================================================||")



# Metaheuristique
best_pattern, best_fobj, iteration_list, rank_list, singular_value_list = metaheuristic_grasp(M, grasp_options)

# Genetic Algorithm
# best_pattern, best_fobj = genetic_algorithm(M, M.shape[0] * M.shape[1], 20000, 0.9999)

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

plot(iteration_list, rank_list, singular_value_list)

