import numpy as np
from objectif_probleme import fobj, compareP1betterthanP2

def generate_initial_pattern(M: np.ndarray) -> np.ndarray:
    """Génère un pattern initial admissible."""
    return np.ones(M.shape)



from typing import Tuple

def metaheuristic(M: np.ndarray, grasp_max_iterations: int=100, grasp_alpha: int=0.1, vns_k_max: int=3) -> Tuple[np.ndarray, Tuple[int, float]]:
    """GRASP complet générant un ensemble de solutions et appliquant une recherche locale."""

    best_pattern = generate_initial_pattern(M)
    best_fobj = fobj(M, best_pattern)
    




    return best_pattern, best_fobj