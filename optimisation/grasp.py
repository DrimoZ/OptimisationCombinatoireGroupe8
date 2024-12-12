from numpy import ndarray, random, array, array_equal, ones

from utils.problem_objective import fobj, compareP1betterthanP2
from optimisation.variable_neighborhood_search import metaheuristic_vns
from optimisation.local_search import local_search

def construct_grasp_solution(M: ndarray, alpha: float, taboo_list: list[ndarray] = []) -> ndarray:
    """
    Construction d'un pattern en utilisant la méthode GRASP.
    
    La construction d'un pattern est réalisée en évaluant les choix possibles 
    pour chaque élément du pattern, puis en sélectionnant un choix probabiliste 
    biaisé basé sur une liste restreinte de candidats.
    
    :param M: Matrice cible
    :type M: np.ndarray
    
    :param alpha: Facteur de construction de la RCL
    :type alpha: float
    
    :param taboo_list: Liste des patterns tabous (déjà explorés et évalués)
    :type taboo_list: list[np.ndarray]
    
    :return: Pattern construit
    :rtype: np.ndarray
    """

    # Initialisation de la matrice P avec des valeurs aléatoires
    m, n = M.shape
    P = random.choice([-1, 1], size=(m, n))
    # P = np.ones((m, n))
    
    # Initialisation de l'ordre de parcours des éléments de P
    order: list = []
    for i in range(m):
        for j in range(n):
            order.append([i, j])

    # Mélanger l'ordre de parcours
    random.shuffle(order)  
    
    # Suppression de X elements de la liste
    # order = order[:int(len(order) * 0.5)] 

    # Parcours des éléments de P
    for i, j in order: 
        
        # Évaluer les choix possibles pour P[i, j] : +1 ou -1
        P[i, j] = 1
        rank1, singular1 = fobj(M, P)
        
        P[i, j] = -1
        rank2, singular2 = fobj(M, P)
        
        # Construire une liste restreinte basée sur les rangs et les singularités
        candidates = [
            (1, rank1, singular1),
            (-1, rank2, singular2)
        ]
        # Trier les candidats par leur rang, puis par leur singularité
        candidates.sort(key=lambda x: (x[1], x[2]))
        
        # Construire la RCL (Restricted Candidate List)
        best_rank = candidates[0][1]
        min_rank = alpha * best_rank
        rcl = [c for c in candidates if c[1] >= min_rank]

        # Retrait des solutions déjà testées
        rcl = [c for c in rcl if not any(array_equal(c[0], sol) for sol in taboo_list)]

        # Sélection probabiliste biaisée
        probabilities = array([1.0 / (i + 1) for i in range(len(rcl))])
        probabilities /= probabilities.sum()  # Normalisation des probabilités
        chosen_index = random.choice(len(rcl), p=probabilities)

        # Fixer la meilleure valeur à P[i, j]
        P[i, j] = rcl[chosen_index][0]

    # Retourner le pattern construit
    return P


def metaheuristic_grasp(M: ndarray, grasp_max_iterations: int = 100, grasp_alpha: float = 0.1, vns_k_max: int = 3, vns_time_limit: int = -1) -> tuple[ndarray, tuple[int, float]]:
    """
    GRASP complet générant une solution optimale selon les paramètres donnés.
    
    La métaheuristique GRASP (Greedy Randomized Adaptive Search Procedure)
    est une méthode itérative qui construit des solutions en utilisant une
    approche gloutonne probabiliste. La recherche à voisinage variable est utilisée pour
    améliorer les solutions construites.
    
    :param M: Matrice cible
    :type M: np.ndarray

    :param grasp_max_iterations: Nombre maximum d'itérations
    :type grasp_max_iterations: int

    :param grasp_alpha: Facteur de construction de la RCL
    :type grasp_alpha: float

    :param vns_k_max: Nombre maximum de changements pour la recherche à voisinage variable
    :type vns_k_max: int
    
    :param vns_time_limit: Limite de temps pour la recherche à voisinage variable
    :type vns_time_limit: int

    :return: Meilleur pattern trouvé et son score (rang, valeur singulière) associé
    :rtype: Tuple[np.ndarray, Tuple[int, float]]
    """

    # Initialisation des variables de la métaheuristique
    best_pattern:ndarray = ones(M.shape, dtype=int)
    best_fobj: tuple[int, float] = fobj(M, best_pattern)

    # Definition d'une liste des candidats testés
    taboo_list: list[ndarray] = [best_pattern]

    # Boucle principale de la métaheuristique
    for current_iteration in range(grasp_max_iterations):

        # Construction d'un pattern
        current_pattern: ndarray = construct_grasp_solution(M, grasp_alpha, taboo_list)

        # Vérifier l'unicité des solutions avant de les ajouter
        if not any(array_equal(current_pattern, sol) for sol in taboo_list):
            taboo_list.append(current_pattern)

            # Recherche à voisinnage variable sur le pattern courant
            # local_search_pattern = local_search(M=M, P=current_pattern, taboo_list=taboo_list, max_duration=10_000_000)
            local_search_pattern, fobj_pattern = metaheuristic_vns(M, best_pattern, current_pattern, vns_k_max, vns_time_limit, taboo_list)
            # local_search_pattern = current_pattern

            # Comparaison avec la meilleure solution
            if compareP1betterthanP2(M, local_search_pattern, best_pattern):
                best_pattern = local_search_pattern
                best_fobj = fobj(M, local_search_pattern)

            if len(taboo_list) >= 2 ** (M.shape[0] * M.shape[1]):
                break

    # Console information
    print(f"||\n|| Realised {current_iteration + 1} iterations")
    print(f"|| Taboo list size: {len(taboo_list)}")

    return best_pattern, best_fobj