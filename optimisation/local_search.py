from numpy import ndarray
from copy import deepcopy

from utils.pattern_neighborhood import generate_complete_neighborhood
from utils.problem_objective import compareP1betterthanP2

def local_search(M: ndarray, P: ndarray, k: int = 1, taboo_list: list[ndarray] = [], max_neighborhood_attempts: int = 500) -> ndarray:
    """
    Recherche locale d'un pattern optimal avec gestion d'une liste tabou.

    La recherche locale est une méthode itérative qui explore le voisinage
    d'un pattern initial P pour trouver un pattern optimal. Les patterns
    déjà testés sont ajoutés à la liste tabou pour éviter les redondances.

    :param M: Matrice cible
    :type M: np.ndarray

    :param P: Pattern initial
    :type P: np.ndarray

    :param k: Nombre d'éléments à changer pour générer un voisinage
    :type k: int
    
    :param taboo_list: Liste des patterns tabous (déjà explorés et évalués)
    :type taboo_list: list[np.ndarray]
    
    :param max_neighborhood_attempts: Nombre maximal de tentatives pour générer un voisin
    :type max_neighborhood_attempts: int

    :return: Pattern optimal
    :rtype: np.ndarray
    """
    
    # Initialiser le pattern courant
    current_pattern: ndarray = deepcopy(P)
    improved: bool = True

    # Tant qu'il y a amélioration
    while improved:
        improved = False

        # Générer le voisinage en fonction de k (en excluant les patterns tabous)
        neighborhood: list[ndarray] = generate_complete_neighborhood(current_pattern, k, taboo_list, max_neighborhood_attempts)
        if len(neighborhood) == 0:
            break

        for neighbor in neighborhood:
            # Ajouter le voisin à la liste tabou
            taboo_list.append(neighbor)
            
            # Si le voisin est meilleur, l'utiliser comme nouveau pattern courant
            if compareP1betterthanP2(M, neighbor, current_pattern):
                current_pattern = neighbor
                improved = True
                break

    # Retourner le pattern optimal
    return current_pattern
