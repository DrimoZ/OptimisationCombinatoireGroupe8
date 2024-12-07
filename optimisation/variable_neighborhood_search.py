from numpy import ndarray, ones, array_equal
from time import time

from utils.problem_objective import fobj, compareP1betterthanP2
from utils.pattern_neighborhood import generate_random_neighborhood
from optimisation.local_search import local_search  

# Time 
vns_end_time: int = time() + 10



    
    
    
      

def metaheuristic_vns(M: ndarray, P: ndarray = None, k_max: int = 3, time_limit: int = 10, taboo_list: list[ndarray] = []) -> tuple[ndarray, tuple[int, float]]:
    """
    Métaheuristique de recherche de voisinage variable (VNS).
    
    La métaheuristique VNS est une méthode itérative qui explore
    les voisinages d'un pattern initial pour trouver un pattern optimal.
    
    :param M: Matrice cible
    :type M: np.ndarray
    
    :param P: Pattern initial
    :type P: np.ndarray
    
    :param k_max: Nombre maximal de voisinages à explorer
    :type k_max: int
    
    :param taboo_list: Liste des patterns tabous (déjà explorés et évalués)
    :type taboo_list: list[np.ndarray]
    
    :return: Pattern optimal et son évaluation
    :rtype: tuple[np.ndarray, tuple[int, float]]
    """
    
    # Initialisation du timer (time_limit secondes)
    vns_end_time = time() + time_limit
    starttimestamp = time()

    # Initialisation
    k: int = 1
    k_max_iter: int = k_max
    stagnation: int = 0
    best_pattern: ndarray
    
    if P is not None:
        best_pattern = P
    else:
        best_pattern = ones(M.shape)

    # Tant que le critère d'arrêt n'est pas atteint
    while k <= k_max_iter:
        
        # Shaking : générer une solution aléatoire dans N_k(best_pattern)
        s_prime: ndarray = generate_random_neighborhood(best_pattern, k, taboo_list, 200, vns_end_time)
        
        # Local search : optimiser autour de s_prime
        s_double_prime: ndarray = local_search(M, s_prime, k, taboo_list, 200, vns_end_time)
                
        # Mise à jour si amélioration
        if compareP1betterthanP2(M, s_double_prime, best_pattern):
            best_pattern = s_double_prime
            k = 1  # Revenir au voisinage initial
            stagnation = 0
        else:
            k += 1  # Diversification : passer au voisinage suivant
            stagnation += 1
            
        # Ajouter le pattern courant à la liste tabou
        if not any(array_equal(c, s_double_prime) for c in taboo_list):
            taboo_list.append(s_double_prime)
        
        # Si la liste tabou est pleine, arrêter la recherche
        if len(taboo_list) >= 2 ** (M.shape[0] * M.shape[1]):
            break
        
        # Si 10 secondes se sont écoulées, arrêter la recherche
        if time() >= vns_end_time:
            break

        # Ajuster dynamiquement k_max
        if stagnation > 5:  # Si aucune amélioration pendant 5 voisinages
            k_max_iter += 1  # Augmenter k_max pour explorer plus loin
        elif stagnation == 0:
            k_max_iter = max(k_max, k_max_iter - 1)  # Réduire k_max si les améliorations sont fréquentes

    return best_pattern, fobj(M, best_pattern)