from typing import Tuple
from datetime import *

import numpy as np


from objectif_probleme import fobj, compareP1betterthanP2
from vns import metaheuristic_vns

# Génération du voisinage (tous les patterns possibles en changeant un seul élément) => 2^n
def generate_neighborhood(P: np.ndarray) -> list[np.ndarray]:
    """Génère le voisinage immédiat d'un pattern."""

    neighbors = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            neighbor = P.copy()
            neighbor[i, j] *= -1  # Flip 1 <-> -1
            neighbors.append(neighbor)
    return neighbors

# Phase de recherche locale
def local_search(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    current_pattern = P.copy()
    improved = True
    while improved:
        improved = False
        neighborhood = generate_neighborhood(current_pattern)
        for neighbor in neighborhood:
            if compareP1betterthanP2(M, neighbor, current_pattern):
                current_pattern = neighbor
                improved = True
                break
    return current_pattern

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

def construct_grasp_solution(taboo_list: list[np.ndarray], M: np.ndarray, alpha: float) -> np.ndarray:
    """
    Construction gloutonne et randomisée d'une matrice P.
    """


    m, n = M.shape
    P = np.random.choice([-1, 1], size=(m, n))  # Initialisation aléatoire
    # P = np.ones((m, n))  # Initialisation de P avec des valeurs par défaut (+1)
    
    order: list = []
    for i in range(m):
        for j in range(n):
            order.append([i, j])

    np.random.shuffle(order)  
    
    # Suppression de X elements de la liste
    # order = order[:int(len(order) * 0.5)] 

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
            candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            
            # Construire la RCL (Restricted Candidate List)
            best_rank = candidates[0][1]
            min_rank = alpha * best_rank
            rcl = [c for c in candidates if c[1] >= min_rank]

            # Retrait des solutions déjà testées
            rcl = [c for c in rcl if not any(np.array_equal(c[0], sol) for sol in taboo_list)]

            # Sélection probabiliste biaisée
            probabilities = np.array([1.0 / (i + 1) for i in range(len(rcl))])
            probabilities /= probabilities.sum()  # Normalisation des probabilités
            chosen_index = np.random.choice(len(rcl), p=probabilities)

            # Fixer la meilleure valeur à P[i, j]
            P[i, j] = rcl[chosen_index][0]
            
    return P



def metaheuristic_grasp(M: np.ndarray, grasp_max_iterations: int=100, grasp_alpha: float=0.1, vns_k_max: int=3) -> Tuple[np.ndarray, Tuple[int, float]]:
    """
    GRASP complet générant une solution optimale selon les paramètres donnés.

    :param M: Matrice cible
    :type M: np.ndarray

    :param grasp_max_iterations: Nombre maximum d'itérations
    :type grasp_max_iterations: int

    :param grasp_alpha: Facteur de construction de la RCL
    :type grasp_alpha: float

    :param vns_k_max: Nombre maximum de changements pour la recherche à voisinage variable
    :type vns_k_max: int

    :return: Meilleur pattern trouvé et son score (rang, valeur singulière) associé
    :rtype: Tuple[np.ndarray, Tuple[int, float]]
    """

    # Initialisation des variables de la métaheuristique
    best_pattern:np.ndarray = np.ones(M.shape, dtype=int)
    best_fobj: Tuple[int, float] = fobj(M, best_pattern)

    # Definition d'une liste des candidats testés
    taboo_list: list[np.ndarray] = []


    for current_iteration in range(grasp_max_iterations):

        # Construction d'un pattern
        current_pattern: np.ndarray = construct_grasp_solution(taboo_list, M, grasp_alpha)

        # Vérifier l'unicité des solutions avant de les ajouter
        if not any(np.array_equal(current_pattern, sol) for sol in taboo_list):
            taboo_list.append(current_pattern)

            # Recherche à voisinnage variable sur le pattern courant
            local_search_pattern, fobj_pattern = metaheuristic_vns(M, current_pattern, vns_k_max)

            # Comparaison avec la meilleure solution
            if compareP1betterthanP2(M, local_search_pattern, best_pattern):
                best_pattern = local_search_pattern
                best_fobj = fobj_pattern

            if taboo_list.__len__() >= 2 ** (M.shape[0] * M.shape[1]):
                break

    # Console information
    print(f"||\n|| Realised {current_iteration + 1} iterations")
    print(f"|| Taboo list size: {len(taboo_list)}")

    return best_pattern, best_fobj