from numpy import ndarray, linalg, sqrt, finfo, where

def fobj(M: ndarray, P: ndarray) -> tuple[int, float]:
    """
    Calcul des objectifs associés à un pattern P.

    :param M: Matrice cible
    :type M: np.ndarray

    :param P: Pattern à évaluer
    :type P: np.ndarray

    :return: Tuple contenant le rang et la plus petite valeur singulière non-nulle de P*sqrt(M)
    :rtype: Tuple[int, float]
    """

    sing_values = linalg.svd(P * sqrt(M), compute_uv = False)       # Calcul des valeurs singulières de la matrice P.*sqrt(M)
    tol         = max(M.shape) * sing_values[0] * finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
    ind_nonzero = where(sing_values > tol)[0]                       # indices des valeurs > tolérance
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]           # outputs: objectif1=rang, objectif2=plus petite val sing. non-nulle


def compareP1betterthanP2(M: ndarray, P1: ndarray, P2: ndarray) -> bool:
    """
    Comparaison de deux patterns P1 et P2.

    :param M: Matrice cible
    :type M: np.ndarray

    :param P1: Premier pattern
    :type P1: np.ndarray

    :param P2: Deuxième pattern
    :type P2: np.ndarray

    :return: Vrai si P1 est meilleur que P2, faux sinon
    :rtype: bool
    """

    r1, s1 = fobj(M, P1)    # on récupère les deux objectifs pour le pattern P1
    r2, s2 = fobj(M, P2)    # on récupère les deux objectifs pour le pattern P2
    if r1 != r2:            # on traite les objectifs de façon lexicographique :
        return r1 < r2      # d'abord les valeurs du rang, et si celles-ci sont égales
    return s1 < s2          # alors on prend en compte la valeur de la + petite valeur singulière