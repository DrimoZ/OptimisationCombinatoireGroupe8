import numpy as np
import random
import time

"""
Fonction de base
"""
# Matrice M générée avec matrices1_ledm
def matrices1_ledm(n):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (i - j)**2
    return M

def fobj(M, P):
    sing_values = np.linalg.svd(P * np.sqrt(M), compute_uv=False)  # Calcul des valeurs singulières de la matrice P.*sqrt(M)
    tol = max(M.shape)*sing_values[0]*np.finfo(float).eps
    ind_nonzero = np.where(sing_values > tol)[0]                    # indices des valeurs > tolérance donnée
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]           # on retourne objectif1=rang et objectif2=plus petite val sing. non-nulle

def compareP1betterthanP2(M, P1, P2):
    """
    Compare les matrices P1 et P2 en fonction des objectifs définis par fobj.
    Les matrices P1 et P2 doivent être binaires (1 et -1).
    """
    r1, s1 = fobj(M, P1)
    r2, s2 = fobj(M, P2)
    if r1 != r2:
        return r1 < r2  # Comparer d'abord les rangs
    return s1 < s2  # Si les rangs sont égaux, comparer les plus petites valeurs singulières

"""
Fonction pour diversité
"""
# voisin alea d'un pattern
def shaking(M, pattern, k):
    """
    Génère une solution aléatoire dans le voisinage N_k(pattern).
    - M : Matrice cible.
    - pattern : Solution actuelle (matrice de -1 et 1).
    - k : Taille du voisinage (nombre de modifications).
    """
    new_pattern = pattern.copy()
    n, m = pattern.shape

    # Sélectionner des indices à modifier avec un mélange aléatoire et biaisé
    indices = [(i, j) for i in range(n) for j in range(m)]

    # Pondérer les indices en fonction de leur impact sur les objectifs
    scores = np.abs(M * new_pattern)  # Exemple : pondération par l'impact local
    scores_flat = scores.flatten()
    probs = scores_flat / np.sum(scores_flat)  # Probabilités normalisées

    # Tirer des indices proportionnels à leur score (avec remplacement pour éviter trop de conflits)
    selected_indices = np.random.choice(range(len(indices)), size=k, p=probs, replace=False)
    
    # Modifier les indices sélectionnés
    for idx in selected_indices:
        i, j = indices[idx]
        new_pattern[i, j] *= -1  # Inversion du signe

    return new_pattern

def generate_all_neighbors(pattern, num_changes, percentage=1):
    """Génère tous les voisins possibles en modifiant `num_changes` valeurs dans le pattern."""
    from itertools import combinations
    import copy

    n, m = pattern.shape
    neighbors = []
    indices = [(i, j) for i in range(n) for j in range(m)]  # Tous les indices possibles
    all_combinations = list(combinations(indices, num_changes))

    # Calculer le nombre de voisins à générer (80 % des voisins possibles)
    num_neighbors_to_generate = int(len(all_combinations) * percentage)
    # Sélectionner un sous-ensemble aléatoire des combinaisons
    selected_combinations = random.sample(all_combinations, num_neighbors_to_generate)

    for combination in selected_combinations:
        new_pattern = copy.deepcopy(pattern)
        for i, j in combination:
            new_pattern[i, j] *= -1  # Inverser le signe

        neighbors.append(new_pattern)

    return neighbors


def local_search(M, current_pattern):
    """
    Recherche locale systématique autour d'une solution.
    - M : Matrice cible
    - current_pattern : Solution actuelle
    - k : Taille du voisinage
    """
    best_pattern = current_pattern
    best_r, best_s = fobj(M, best_pattern)

    neighbors = generate_all_neighbors(current_pattern, num_changes=2)  
    for neighbor in neighbors:
        if compareP1betterthanP2(M, neighbor, best_pattern):
            best_pattern = neighbor
            best_r, best_s = fobj(M, best_pattern)

    return best_pattern, best_r, best_s

"""
Fonction algo
"""
def metaheuristic_vns(M,best_pattern):
    """
    Variable Neighborhood Search (VNS) pour minimiser les objectifs lexicographiques.
    - M : Matrice cible
    - max_iterations : Nombre maximal d'itérations globales
    """
    best_r, best_s = fobj(M, best_pattern)
    k = 1 
    stagnation = 0
    last_r_double_prime = 0
    same_rank= 0

    while k <= 2:
        s_prime = shaking(M,best_pattern,5)
        s_double_prime, r_double_prime, s_double_s = local_search(M, s_prime)

        if last_r_double_prime == r_double_prime:
            same_rank += 1
            if same_rank == 500:
                print('end because of same rank')
                break
        last_r_double_prime = r_double_prime

        # Mise à jour si amélioration
        if compareP1betterthanP2(M, s_double_prime, best_pattern):
            best_pattern = s_double_prime
            best_r, best_s = r_double_prime, s_double_s 
            stagnation = 0
            print(best_r)
        else:
            stagnation += 1

        if stagnation >= 5:
            shaked_p = shaking(M,best_pattern,5)
            shaked_p, r_double_prime, s_double_s = local_search(M, shaked_p)
            if compareP1betterthanP2(M,shaked_p, best_pattern):
                best_pattern = shaked_p
                stagnation = 0
        else:
            stagnation +=1

        if best_r == 2:
            break

    return best_pattern, (best_r, best_s)

def simulated_annealing(M, P, T=10000.0, cooling_rate=0.9995, max_iter=5000):
    """
    Applique une phase de recuit simulé pour explorer des solutions voisines.
    """
    current_P = P.copy()
    current_rank, current_singular = fobj(M, current_P)
    best_P = current_P
    best_rank, best_singular = current_rank, current_singular

    for _ in range(max_iter):
        # Générer un voisin aléatoire
        neighbor= shaking(M,current_P, 1)
        neighbor_rank, neighbor_singular = fobj(M,neighbor)
        # Calculer la différence de coût
        delta = neighbor_rank - current_rank
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_P = neighbor
            current_rank, current_singular = neighbor_rank, neighbor_singular
            if compareP1betterthanP2(M, current_P, best_P):
                best_P = current_P
                best_rank, best_singular = current_rank, current_singular

        # Refroidir la température
        T *= cooling_rate

        if T <=1e-3:
            break

    return best_P, best_rank, best_singular

def tabu_search_hybrid(M, P, n,tabu_list, max_iter=5000, tabu_size=2000, neighborhood_size=12):
    current_rank, current_smallest_singular = fobj(M, P)
    best_P = P.copy()
    best_rank, best_smallest_singular = current_rank, current_smallest_singular

    inferior = 0
    stagnation_count = 0

    for iteration in range(max_iter):
        neighbors = []
        for _ in range(neighborhood_size):
            neighbor = P.copy()
            x1, x2 = sorted(np.random.randint(0, n, size=2))
            y1, y2 = sorted(np.random.randint(0, n, size=2))
            neighbor[x1:x2+1, y1:y2+1] *= -1
            neighbors.append(neighbor)

        best_neighbor = None
        best_neighbor_rank = np.inf
        best_neighbor_singular = np.inf

        for neighbor in neighbors:
            if tuple(map(tuple, neighbor)) in tabu_list:
                continue

            rank, smallest_singular = fobj(M, neighbor)

            if rank < best_neighbor_rank or (
                rank == best_neighbor_rank and smallest_singular < best_neighbor_singular
            ):
                best_neighbor = neighbor
                best_neighbor_rank = rank
                best_neighbor_singular = smallest_singular

        if best_neighbor is None:
            P = np.random.choice([-1, 1], size=(n, n))
            current_rank, current_smallest_singular = fobj(M, P)
            continue

        if inferior >= 50:
            # Phase de shaking : Appliquer une perturbation significative
            P = shaking(M,P, 5)
            current_rank, current_smallest_singular = fobj(M, P)
            inferior = 0
            stagnation_count += 1

            # Phase hybride : Lancer le recuit simulé
            if stagnation_count >= 3:
                P, current_rank, current_smallest_singular = simulated_annealing(M, P)
                P, current_rank, current_smallest_singular = local_search(M, P)
                
            if stagnation_count == 50:
                break

        P = best_neighbor
        current_rank, current_smallest_singular = best_neighbor_rank, best_neighbor_singular

        if compareP1betterthanP2(M, P, best_P):
            best_P = P.copy()
            best_rank, best_smallest_singular = current_rank, current_smallest_singular
            tabu_list.append(tuple(map(tuple, P)))

            # Vider la moitié de la liste tabou si elle atteint son maximum
            if len(tabu_list) == tabu_size:
                for _ in range(tabu_size // 2):
                    tabu_list.popleft()
        else:
            inferior += 1

        # Réduire dynamiquement la taille du voisinage si le progrès stagne
        if iteration % 1000 == 0 and inferior >= 20:
            neighborhood_size = max(5, neighborhood_size - 1)

        if best_rank == 2:
            break

    return best_P, (best_rank, best_smallest_singular), tabu_list

"""
Déclaration
"""
n = 120
M = matrices1_ledm(n)
tabu_list = []
P = np.ones(M.shape)

"""
Run & Result
"""
start_time = time.time()
for pass_num in range(20):
    print(f"Passage {pass_num + 1} :")

    print('VNS')
    P, rank = metaheuristic_vns(M,P)
    print(f"Rank : {rank}")

    print('TABOO')
    P, rank,tabu_list = tabu_search_hybrid(M, P, n, tabu_list) 
    print(f'rank {rank[0]} , smallest_singular : {rank[1]}') 

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Temps d'exécution : {elapsed_time:.4f} secondes")
print(f'smallest rank {rank[0]}, smallest singular : {rank[1]}\n with P :\n {P} \nfor \n{M}')