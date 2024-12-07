import numpy as np
from scipy.linalg import circulant

def matrices1_ledm(n: int) -> np.ndarray:
    """
    Génère une matrice de distance euclidienne de taille n x n.
    
    :param n: Taille de la matrice
    :type n: int
    
    :return: Matrice de distance euclidienne
    :rtype: np.ndarray
    """
    
    M: np.ndarray = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            M[i, j]=(i - j) ** 2
            
    return M

def matrices2_slackngon(n: int)-> np.ndarray:
    """
    Génère une matrice de slackngon
    
    :param n: Taille de la matrice
    :type n: int
    
    :return: Matrice de slackngon
    :rtype: np.ndarray
    """
    
    M: np.ndarray = circulant(np.cos(np.pi / n) - np.cos(np.pi / n + 2 * np.pi * np.arange(0, n, 1) / n))
    M /= M[0, 2]
    M  = np.maximum(M, 0)
    
    for i in range(n):
        M[i, i] = 0
        if i < n - 1:
            M[i, i + 1] = 0
        else:
            M[i, 0] = 0
            
    return M