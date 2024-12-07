def generate_initial_pattern(M: np.ndarray) -> np.ndarray:
    """
    Génère un pattern initial admissible de même taille que M.

    Le pattern initial est un pattern où tous les éléments sont égaux à 1.

    :param M: Matrice cible
    :type M: np.ndarray

    :return: Pattern initial
    :rtype: np.ndarray
    """
    
    return np.ones(M.shape, dtype=int)

def generate_random_pattern(M: np.ndarray) -> np.ndarray:
    """
    Génère un pattern admissible aléatoire de même taille que M.

    :param M: Matrice cible
    :type M: np.ndarray

    :return: Pattern aléatoire
    :rtype: np.ndarray
    """

    n: int = M.shape[0]
    P: np.ndarray = np.ones((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if np.random.random() < 0.5:
                P[i, j] = -1

    return P

def generate_rcl(M: np.ndarray, alpha: float) -> list[np.ndarray]:
    """Génère une liste restreinte de candidats."""
    
    # Calcul de la valeur minimale et maximale de la matrice
    min_value: int = np.min(M)
    max_value: int = np.max(M)
    
    # Calcul de la valeur seuil
    threshold: float = min_value + alpha * (max_value - min_value)

    # Génération de patterns pour chaque élément de la RCL
    rcl_patterns: list[np.ndarray] = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] <= threshold:
                pattern = np.ones(M.shape) * -1
                pattern[i, j] = 1
                rcl_patterns.append(pattern)

    return rcl_patterns

def construct_rcl_patterns(M: np.ndarray, alpha: float, best_actual_pattern: np.ndarray, num_solutions: int = 10) -> list[np.ndarray]:
    """Phase de construction générant un ensemble de solutions."""
    n: int = M.shape[0]
    solutions: list[np.ndarray] = []
    
    for _ in range(num_solutions):
        # Point de départ légèrement perturbé pour chaque solution
        P = best_actual_pattern.copy()
        for _ in range(np.random.randint(1, 5)):
            i, j = np.random.randint(n), np.random.randint(n)
            P[i, j] = -P[i, j]
        
        for i in range(n):
            for j in range(n):
                # Générer des candidats en modifiant un élément
                candidates = []
                for value in [-1, 1]:
                    if P[i, j] != value:
                        candidate = P.copy()
                        candidate[i, j] = value
                        candidates.append(candidate)
                
                # Calculer les scores des candidats
                candidate_scores = [
                    (candidate, fobj(M, candidate))
                    for candidate in candidates
                ]
                candidate_scores.sort(key=lambda x: (x[1][0], x[1][1]))  # Trier par rang, puis valeur singulière
                
                # Définir les seuils pour la RCL
                best_score = candidate_scores[0][1]
                worst_score = candidate_scores[-1][1]
                threshold_rank = best_score[0] + alpha * (worst_score[0] - best_score[0])
                
                # Construire la RCL
                RCL = [
                    candidate
                    for candidate, score in candidate_scores
                    if score[0] <= threshold_rank
                ]
                
                # Sélection aléatoire dans la RCL
                if len(RCL) > 0:
                    P = RCL[np.random.randint(len(RCL))]
        
        # Vérifier l'unicité des solutions avant de les ajouter
        if not any(np.array_equal(P, sol) for sol in solutions):
            solutions.append(P)
    
    return solutions
