from numpy import ndarray, array_equal, random
from itertools import combinations
from copy import deepcopy

def generate_random_neighborhood(P: ndarray, k: int = 1, taboo_list: list[ndarray] = []) -> ndarray:
    """
    Génère un voisin aléatoire d'un pattern.
    
    Un voisin est obtenu en inversant k valeurs dans le pattern initial.
    
    :param pattern: Pattern initial
    :type pattern: np.ndarray
    
    :param k: Nombre de valeurs à inverser
    :type k: int
    
    :param taboo_list: Liste des patterns tabous (déjà explorés et évalués)
    :type taboo_list: list[np.ndarray]
    
    :return: Voisin aléatoire
    :rtype np.ndarray
    """

    n, m = P.shape

    while True:
        # Créer une copie du pattern initial
        new_pattern: ndarray = deepcopy(P)
        
        # Inverser k valeurs aléatoires
        for _ in range(k):
            i: int = random.randint(0, n) 
            j: int = random.randint(0, m)
            
            new_pattern[i, j] *= -1

        # Retourner le voisin si ce n'est pas un pattern tabou
        if not any(array_equal(c, new_pattern) for c in taboo_list):
            return new_pattern


def generate_complete_neighborhood(P: ndarray, k: int = 1,  taboo_list: list[ndarray] = []) -> list[ndarray]:
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

    :return: Voisinage complet de P
    :rtype: list[np.ndarray]
    """

    # Initialiser la liste des voisins
    neighbors: list[ndarray] = []

    # Obtenir tous les indices de P sous forme de liste de tuples
    indices: list[tuple[int, int]] = [(i, j) for i in range(P.shape[0]) for j in range(P.shape[1])]

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

    # Retourner la liste des voisins
    return neighbors