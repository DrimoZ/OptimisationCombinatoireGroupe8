from datetime import datetime, timedelta
from numpy import ndarray, random, loadtxt, logspace, linspace
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

# Function to plot the evolution of scores as a function of alpha
def plot_alpha_vs_scores(alpha_values, scores):
    """
    Plots the evolution of the score (rank + lambda_ * singular_value)
    as a function of alpha values.
    
    :param alpha_values: List of alpha values tested
    :param scores: List of scores corresponding to the alpha values
    """
    plt.figure(figsize=(10, 6))

    plt.plot(alpha_values, scores, marker='o', label='Score', color='purple')
    plt.xlabel("Alpha (α)")
    plt.ylabel("Score")
    plt.title("Score Evolution as a Function of Alpha")
    plt.xscale('log')
    plt.grid()
    plt.legend()

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


# Alpha values to test
alpha_values = linspace(0.000001, 1, num=2000)  # Evenly spaced alpha values from 0.0001 to 1

# Container for scores
scores = []

# Temps de début
starttime: datetime = datetime.now()
print("\n\n||================================ GRASP ================================||")
print(f"|| Grasp started on {starttime.date()} at {starttime.time()}                        ||")
print("||=======================================================================||")

# Loop over alpha values
# for alpha in alpha_values:
#     print(f"Testing alpha = {alpha:.4f}")
#     grasp_options.alpha = alpha  # Update alpha in options

#     # Run GRASP metaheuristic
#     best_pattern, best_fobj, _, _, _ = metaheuristic_grasp(M, grasp_options)

#     # Calculate the score
#     rank, smallest_sv = best_fobj
#     score = rank + 0.03 * smallest_sv
#     scores.append(score)


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

# Plot results
plot_alpha_vs_scores(alpha_values, scores)