from numpy import ndarray, random, array, array_equal, ones
from random import choices

from utils.orthogonal_matrix import matrices1_ledm, matrices2_slackngon
from utils.problem_objective import fobj

from utils.pattern_neighborhood import generate_complete_neighborhood

def generate_population(size: int, m: int, n: int) -> list[ndarray]:
    """
    Génère une population de patterns aléatoires.
    
    :param size: Taille de la population
    :type size: int
    
    :param m: Nombre de lignes d'un pattern
    :type m: int
    
    :param n: Nombre de colonnes d'un pattern
    :type n: int
    
    :return: Population de pattern aléatoires
    :rtype: list[np.ndarray]
    """
    
    # Eviter les doublons
    population = set()
    
    while len(population) < size:
        pattern = random.choice([-1, 1], size=(m, n))
        population.add(pattern)
    
    return [array(p).reshape(m, n) for p in population]

def evaluate_population(M: ndarray, population: list[ndarray]) -> list[tuple[tuple[int, float], ndarray]]:
    """
    Évalue la fitness de chaque individu dans la population.
    """
    return [(fobj(M, P), P) for P in population]

def select_parents(population_fitness: list[tuple[tuple[int, float], ndarray]]) -> tuple[ndarray, ndarray]:
    """
    Sélectionne deux parents (tournoi ou roulette).
    
    :param population_fitness: Population triée par fitness
    :type population_fitness: list[tuple[tuple[int, float], np.ndarray]]
    
    :return: Deux parents sélectionnés
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    
    # Trier par fitness (meilleur rang puis plus petite singularité)
    population_fitness.sort(key=lambda x: (x[0][0], x[0][1]))
    
    return choices(population_fitness[:len(population_fitness)//2], k=2)

def crossover(parent1: ndarray, parent2: ndarray) -> ndarray:
    """
    Réalise un croisement entre deux parents.
    
    :param parent1: Premier parent
    :type parent1: np.ndarray
    
    :param parent2: Deuxième parent
    :type parent2: np.ndarray
    
    :return: Enfant généré
    :rtype: np.ndarray
    """
    m, n = parent1.shape
    child = parent1.copy()
    for i in range(m):
        for j in range(n):
            if random.random() < 0.5:
                child[i, j] = parent2[i, j]
    return child

def mutate(pattern: ndarray, mutation_rate: float = 0.01):
    """
    Applique une mutation au pattern donné.
    
    :param matrix: Pattern à muter
    :type matrix: np.ndarray
    
    :param mutation_rate: Taux de mutation
    :type mutation_rate: float
    
    :return: Pattern muté
    :rtype: np.ndarray
    """
    m, n = pattern.shape
    for i in range(m):
        for j in range(n):
            if random.random() < mutation_rate:
                pattern[i, j] *= -1  # Inverse le signe
    return pattern

def genetic_algorithm(M, population_size=50, generations=100, mutation_rate=0.01):
    """
    Algorithme génétique principal.
    """
    
    m, n = M.shape
    
    # Générer la population initiale
    population: list[ndarray] = generate_population(population_size, m, n)
    
    for gen in range(generations):
        # Évaluation de la population
        population_fitness = evaluate_population(M, population)
        
        # Sélection des meilleurs individus
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population_fitness)
            parent1, parent2 = parents[0][1], parents[1][1]
            
            # Croisement
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            
            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Mise à jour de la population
        population = new_population
        
        # Meilleur individu de la génération
        best_individual = max(population_fitness, key=lambda x: (x[0][0], x[0][1]))
        
        if best_individual[0][0] <= 2:
            break

    # Retourner le meilleur individu
    return best_individual[1], best_individual[0]
