from numpy import ndarray, array_equal, random
from itertools import combinations
from copy import deepcopy
from time import time

def generate_random_neighborhood(P: ndarray, k: int = 1, taboo_list: list[ndarray] = [], max_attempts: int = 1000, vns_end_time: int = -1) -> ndarray | None:
    """
    Génère un voisin aléatoire d'un pattern.
    
    Un voisin est obtenu en inversant k valeurs dans le pattern initial.
    
    :param pattern: Pattern initial
    :type pattern: np.ndarray
    
    :param k: Nombre de valeurs à inverser
    :type k: int
    
    :param taboo_list: Liste des patterns tabous (déjà explorés et évalués)
    :type taboo_list: list[np.ndarray]
    
    :param max_attempts: Nombre maximal de tentatives pour générer un voisin
    :type max_attempts: int
    
    :param vns_end_time: Temps de fin de la métaheuristique VNS (uniquement pour utilisation avec VNS)
    :type vns_end_time: int
    
    :return: Voisin aléatoire de pattern ou None si aucun voisin n'est trouvé
    :rtype: np.ndarray
    """

    n, m = P.shape

    for _ in range(max_attempts):
        # Créer une copie du pattern initial
        new_pattern: ndarray = deepcopy(P)
        
        # Inverser k valeurs aléatoires
        for _ in range(k):
            i: int = random.randint(0, n - 1)  # Assurez-vous que les indices sont valides
            j: int = random.randint(0, m - 1)
            
            new_pattern[i, j] *= -1

        # Retourner le voisin si ce n'est pas un pattern tabou
        if not any(array_equal(c, new_pattern) for c in taboo_list):
            return new_pattern
        
        # Arrêter si le temps de fin est atteint
        if vns_end_time > 0 and time() > vns_end_time:
            break

    # Si aucune solution n'est trouvée après max_attempts, retourner None
    return P


def generate_complete_neighborhood(P: ndarray, k: int = 1,  taboo_list: list[ndarray] = [], max_attempts: int = 1000) -> list[ndarray]:
    """
    Génère le voisinage courrant complet d'un pattern P.

    Le voisinage complet d'un pattern P est l'ensemble des patterns
    obtenus en changeant un unique élément de P.

    :param P: Pattern initial
    :type P: np.ndarray
    
    :param k: Nombre d'éléments à changer pour générer le voisinage
    :type k: int

    :param taboo_list: Liste des patterns tabous (déjà explorés et evalués)
    :type taboo_list: list[np.ndarray]
    
    :param max_attempts: Nombre maximal de tentatives pour générer un voisin
    :type max_attempts: int

    :return: Voisinage complet de P
    :rtype: list[np.ndarray]
    """

    # Initialiser la liste des voisins
    neighbors: list[ndarray] = []

    # Obtenir tous les indices de P sous forme de liste de tuples
    indices: list[tuple[int, int]] = [(i, j) for i in range(P.shape[0]) for j in range(P.shape[1])]
    random.shuffle(indices)

    # Initialiser l'itération
    iteration: int = 0
    
    # Générer toutes les combinaisons possibles de k indices
    for combo in combinations(indices, k):
        
        # Créer une copie du pattern original
        neighbor: ndarray = deepcopy(P)

        # Appliquer les changements pour les k indices sélectionnés
        for (i, j) in combo:
            neighbor[i, j] *= -1

        # Ajouter le voisin si ce n'est pas un pattern tabou
        if not any(array_equal(c, neighbor) for c in taboo_list):
            neighbors.append(neighbor)
            
        # Incrémenter l'itération
        iteration += 1

        # Arrêter si le nombre maximal d'itérations est atteint       
        if iteration >= max_attempts:
            break

    # Retourner la liste des voisins
    return neighbors