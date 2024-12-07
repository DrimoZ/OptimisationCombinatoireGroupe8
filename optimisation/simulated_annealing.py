import numpy as np
import matplotlib.pyplot as plt

def matrices1_ledm(n):
  M  = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      M[i,j]=(i-j)**2
  return M

from scipy.linalg import circulant
def matrices2_slackngon(n):
  M  = circulant(np.cos(np.pi/n)-np.cos(np.pi/n + 2*np.pi*np.arange(0,n,1)/n))
  M /= M[0,2]
  M  = np.maximum(M,0)
  for i in range(n):
    M[i,i] = 0
    if i<n-1:
      M[i,i+1] = 0
    else:
      M[i,0] = 0
  return M

def fobj(M,P,tol=1e-14):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False) # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                # indices des valeurs > tolérance donnée
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]       # on retourne objectif1=rang et objectif2=plus petite val sing. non-nulle

def compareP1betterthanP2(M,P1,P2):
  r1, s1 = fobj(M,P1) #on récupère les deux objectifs pour le pattern P1
  r2, s2 = fobj(M,P2) #on récupère les deux objectifs pour le pattern P2
  if r1 != r2:       #on traite les objectifs de façon lexicographique :
      return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
  return s1 < s2      # alors on prend en compte la valeur de la + petite valeur singulière

def recuit_simule(M, init_pattern, max_iter=150000, T_init=10000000000, alpha=0.99, tol=1e-14):
    """
    Algorithme de recuit simulé pour trouver le meilleur pattern P.
    
    :param M: Matrice donnée
    :param init_pattern: Pattern initial
    :param max_iter: Nombre maximum d'itérations
    :param T_init: Température initiale
    :param alpha: Facteur de refroidissement
    :param tol: Tolérance pour les valeurs singulières
    :return: Meilleur pattern trouvé
    """
 
    def generate_voisin(P, n_changes=2):
        P_voisin = P.copy()
        for _ in range(n_changes):
          i, j = np.random.randint(0, P.shape[0]), np.random.randint(0, P.shape[1])
          P_voisin[i, j] *= -1
        return P_voisin


    # Initialisation
    current_pattern = init_pattern
    best_pattern = init_pattern
    current_obj = fobj(M, current_pattern)
    best_obj = current_obj
    T = T_init
    i = 0
    non = 0
    oui = 0

    # Initialisation de la figure pour les graphiques
    plt.figure(figsize=(10, 5))
    rangs = []
    min_sing_values = []
    
    for _ in range(max_iter):
        i += 1
        # Générer un voisin aléatoire
        new_pattern = generate_voisin(current_pattern)
        new_obj = fobj(M, new_pattern)

        # Calcul de la variation d'énergie
        delta_E = (new_obj[0] - current_obj[0]) if new_obj[0] != current_obj[0] else (new_obj[1] - current_obj[1])

        # Décision d'accepter ou non le voisin
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_pattern = new_pattern
            current_obj = new_obj
            oui += 1

            # Mettre à jour le meilleur pattern trouvé
            if compareP1betterthanP2(M, new_pattern, best_pattern):
                best_pattern = new_pattern
                best_obj = new_obj

        else:
           non += 1

        # Réduire la température
        T *= alpha

        # Ajouter les objectifs à la liste pour affichage graphique
        rangs.append(current_obj[0])
        min_sing_values.append(current_obj[1])

        # Affichage des graphiques à intervalles réguliers
        if i % 100 == 0 or i == max_iter-1:
            plt.clf()  # Effacer la figure précédente
            plt.subplot(1, 2, 1)
            plt.plot(rangs, label='Rang')
            plt.xlabel('Itérations')
            plt.ylabel('Rang')
            plt.title('Évolution du Rang')

            plt.subplot(1, 2, 2)
            plt.plot(min_sing_values, label='Plus petite valeur singulière', color='red')
            plt.xlabel('Itérations')
            plt.ylabel('Valeur Singulière')
            plt.title('Évolution de la Plus Petite Valeur Singulière')

            plt.tight_layout()
            plt.pause(0.1)  # Mettre en pause pour voir la mise à jour graphique

        # Arrêt si la température devient très faible
        if T < 1e-8:
            break

    print(i)
    print("nb patterns acceptés : ", oui)
    print("nb patterns non-acceptés : ", non)
    
    # Afficher les graphiques finaux
    plt.show()
    return best_pattern




# Générer la matrice M
M = matrices2_slackngon(5)
#M = np.array([[16,4,1,25,4],[16,4,4,36,0],[0,0,9,1,4],[36,9,0,64,4],[16,4,1,25,4],[4,1,49,25,16]])

# Initialiser le pattern
init_pattern = np.ones(M.shape)

# Appliquer le recuit simulé
best_pattern = recuit_simule(M, init_pattern)

print("Meilleur pattern trouvé :")
print(best_pattern)

result = M * best_pattern
print("Produit de la matrice M et du pattern optimal :")
print(result)
r1, s1 = fobj(M,best_pattern)
print("Rang : ", r1)
