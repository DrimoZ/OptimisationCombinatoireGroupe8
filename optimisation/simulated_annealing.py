from numpy import ndarray, random, exp

from utils.pattern_neighborhood import generate_random_neighborhood
from utils.problem_objective import fobj

def simulated_annealing(M, initial_solution, max_iterations=100000, initial_temp=1000000, cooling_rate=0.99):
  current_solution = initial_solution
  current_score = fobj(M, current_solution)
  best_solution = current_solution
  best_score = current_score

  temperature = initial_temp
  for iteration in range(max_iterations):
    next_solution = generate_random_neighborhood(current_solution)
    next_score = fobj(M, next_solution)

    # Calculer la différence de score
    delta_score = next_score[0] - current_score[0]  # Basé sur le rang ici
    if delta_score == 0:  # Si rang identique, on compare les valeurs singulières
      delta_score = next_score[1] - current_score[1]

    # Accepter la solution si elle est meilleure ou avec une probabilité liée à T
    if delta_score < 0 or exp(-delta_score / temperature) > random.random():
      current_solution = next_solution
      current_score = next_score

      # Mettre à jour la meilleure solution
      if next_score < best_score:
        best_solution = next_solution
        best_score = next_score

    if iteration % 10 == 0:
      # Réduire la température
      temperature= cooling_rate

    # Arrêter si la température est trop basse
    if temperature < 1e-5:
      break

  return best_solution