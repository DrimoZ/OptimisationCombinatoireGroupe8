import numpy as np
from objectif_probleme import fobj, compareP1betterthanP2


# voisin alea d'un pattern
def generate_neighborhood(pattern, num_changes=1):
    """Génère un voisin en inversant `num_changes` valeurs dans le pattern."""
    new_pattern = pattern.copy()
    n, m = pattern.shape
    for _ in range(num_changes):
        i, j = np.random.randint(0, n), np.random.randint(0, m)
        new_pattern[i, j] *= -1  # Inverser le signe
    return new_pattern

def generate_all_neighbors(pattern, num_changes):
    """Génère tous les voisins possibles en modifiant `num_changes` valeurs dans le pattern."""
    from itertools import combinations
    import copy

    n, m = pattern.shape
    neighbors = []
    indices = [(i, j) for i in range(n) for j in range(m)]  # Tous les indices possibles

    for combination in combinations(indices, num_changes):
        new_pattern = copy.deepcopy(pattern)
        for i, j in combination:
            new_pattern[i, j] *= -1  # Inverser le signe
            
            #TIME
            if check_time():
              return neighbors

        neighbors.append(new_pattern)

        #TIME
        if check_time():
          return neighbors

    return neighbors


def local_search(M, current_pattern, k):
    """
    Recherche locale systématique autour d'une solution.
    - M : Matrice cible
    - current_pattern : Solution actuelle
    - k : Taille du voisinage
    """
    best_pattern = current_pattern
    best_r, best_s = fobj(M, best_pattern)

    neighbors = generate_all_neighbors(current_pattern, num_changes=k)  # Tous les voisins de `current_pattern`
    for neighbor in neighbors:
        if check_time():
           break
        if compareP1betterthanP2(M, neighbor, best_pattern):
            best_pattern = neighbor
            best_r, best_s = fobj(M, best_pattern)
    
    return best_pattern, best_r, best_s

import time as t

time_data = {
  "start_time": t.time(),  # Heure actuelle (temps de début)
  "current_time": t.time()  # Initialisation de current_time (pas encore modifié)
}
def check_time():
  time_data["current_time"] = t.time()
  if time_data["current_time"] - time_data["start_time"] >= 60 :
    return True
  return False
    
def metaheuristic_vns(M, k_max=3):
  """
  Variable Neighborhood Search (VNS) pour minimiser les objectifs lexicographiques.
  - M : Matrice cible
  - max_iterations : Nombre maximal d'itérations globales
  - k_max : Nombre maximal de voisinages (eleve = diversification / petit intensification)
  """
  # init pattern full +1
  #best_pattern = np.random.choice([-1, 1], M.shape)
  best_pattern=np.ones(M.shape)
  best_r, best_s = fobj(M, best_pattern) 

 
  k = 1  # Démarrer au premier voisinage
  stagnation = 0

  while k <= k_max:
      # Shaking : générer une solution aléatoire dans N_k(best_pattern)
      s_prime = generate_neighborhood(best_pattern, num_changes=k)

      if check_time():
        return best_pattern, (best_r, best_s)
      # Local search : optimiser autour de s_prime
      s_double_prime, r_double_prime, s_double_s = local_search(M, s_prime, k)

      # Mise à jour si amélioration
      if compareP1betterthanP2(M, s_double_prime, best_pattern):
          best_pattern = s_double_prime
          best_r, best_s = r_double_prime, s_double_s
          k = 1  # Revenir au voisinage initial
          stagnation = 0
      else:
          k += 1  # Diversification : passer au voisinage suivant
          stagnation += 1

      #TIME
      if check_time():  # Si 10 secondes sont écoulées
        return best_pattern, (best_r, best_s)

      # Ajuster dynamiquement k_max
      if stagnation > 5:  # Si aucune amélioration pendant 5 voisinages
          k_max += 1  # Augmenter k_max pour explorer plus loin
      elif stagnation == 0:
          k_max = max(3, k_max - 1)  # Réduire k_max si les améliorations sont fréquentes

  return best_pattern, (best_r, best_s)