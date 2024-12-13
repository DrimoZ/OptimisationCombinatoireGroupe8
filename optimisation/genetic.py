import numpy as np

from utils.orthogonal_matrix import matrices1_ledm, matrices2_slackngon
from utils.problem_objective import fobj

def generate_population(size, m, n):
    """
    Génère une population initiale de matrices.
    """
    return [np.random.choice([-1, 1], size=(m, n)) for _ in range(size)]

def evaluate_population(M, population):
    """
    Évalue la fitness de chaque individu dans la population.
    """
    return [(fobj(M, P), P) for P in population]

def select_parents(population_fitness):
    """
    Sélectionne deux parents (tournoi ou roulette).
    """
    # Trier par fitness (meilleur rang puis plus petite singularité)
    population_fitness.sort(key=lambda x: (x[0][0], -x[0][1]), reverse=True)
    return np.random.choice(population_fitness[:len(population_fitness)//2], k=2)

def crossover(parent1, parent2):
    """
    Réalise un croisement entre deux parents.
    """
    m, n = parent1.shape
    child = parent1.copy()
    for i in range(m):
        for j in range(n):
            if np.random.random() < 0.5:
                child[i, j] = parent2[i, j]
    return child

def mutate(matrix, mutation_rate=0.01):
    """
    Applique une mutation à la matrice.
    """
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            if np.random.random() < mutation_rate:
                matrix[i, j] *= -1  # Inverse le signe
    return matrix

def genetic_algorithm(M, population_size=50, generations=100, mutation_rate=0.01):
    """
    Algorithme génétique principal.
    """
    
    m, n = M.shape
    
    # Générer la population initiale
    population = generate_population(population_size, m, n)
    
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
        best_individual = max(population_fitness, key=lambda x: (x[0][0], -x[0][1]))
        print(f"Génération {gen+1} : Rang = {best_individual[0][0]}, Singularité = {best_individual[0][1]}")

    # Retourner le meilleur individu
    return best_individual[1]

# Exemple d'utilisation

M = matrices1_ledm(6)

best_P = genetic_algorithm(M, population_size=50, generations=50, mutation_rate=0.01)

print("Meilleure matrice P trouvée :")
print(best_P)
