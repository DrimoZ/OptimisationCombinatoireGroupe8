from numpy import ndarray, ones, random, inf, arange, copy, put, take
from random import choices

from utils.problem_objective import fobj, compareP1betterthanP2
from options import GraspConstructionOptions, LocalSearchOptions


def pattern_to_tuple(P: ndarray) -> tuple:
    """
    Convert a pattern matrix P (np.ndarray) into a tuple of tuples to make it hashable.
    """
    
    return tuple(tuple(row) for row in P)

def perturb_pattern(P: ndarray, perturbation_rate: float) -> ndarray:
    """
    Perturb (mutate) the given pattern with a certain perturbation rate.
    Changes a random subset of the entries in the pattern.

    :param P: The pattern matrix to perturb.
    :type P: np.ndarray
    :param perturbation_rate: The rate of perturbation, determining how many elements to change.
    :type perturbation_rate: float
    :return: The perturbed pattern.
    :rtype: np.ndarray
    """
    
    n = P.shape[0]
    num_perturb = max(1, int(perturbation_rate * n * n))  # Number of elements to perturb
    perturb_indices = random.choice(n * n, num_perturb, replace=False)
    
    for index in perturb_indices:
        x, y = divmod(index, n)
        P[x, y] = random.choice([1, -1])  # Randomly flip the value at position (x, y)
    
    return P

def grasp_constructive_optimized(M: ndarray, tested_patterns: set[tuple], options: GraspConstructionOptions) -> ndarray:
    """
    Optimized GRASP construction algorithm with minimal randomization and perturbation.

    :param M: Target matrix
    :type M: np.ndarray
    
    :param options: GRASP construction options
    :type options: GraspConstructionOptions
    
    :return: Constructed pattern
    :rtype: np.ndarray
    """
    
    # Pattern initialization
    n, m = M.shape
    P = random.choice([1, -1], size=(n, m))
    # P = np.ones((n, n))  # Initialize pattern with all ones
    
    # Variables initialization
    last_cost = inf
    stagnation_count = 0
    repetition_count = 0
    local_perturbation_rate = options.perturbation_rate

    # Candidate list initialization
    candidate_list = [(i, j) for i in range(n) for j in range(n)]

    # Main Construction loop
    while candidate_list:
        random.shuffle(candidate_list)
        sampled_groups = []
        
        # Random sample groups of candidates
        combination_size = min(len(candidate_list), options.max_fixed_candidates_per_iteration)
        
        while len(candidate_list) >= combination_size:
            group = tuple(candidate_list[:combination_size])
            sampled_groups.append(group)
            candidate_list = candidate_list[combination_size:]

        # Evaluate sampled groups
        costs = []
        for group in sampled_groups:
            best_cost = inf
            best_values = {}
            
            sampled_values = [choices([1, -1], k=len(group)) for _ in range(len(group))]
            
            for value_combination in sampled_values:
                P_temp = copy(P)
                for (i, j), value in zip(group, value_combination):
                    P_temp[i, j] = value

                rank, inv_smallest_sv = fobj(M, P_temp)
                cost = rank + options.lambda_ * inv_smallest_sv

                if cost < best_cost:
                    best_cost = cost
                    best_values = {pos: val for pos, val in zip(group, value_combination)}
                

            costs.append((group, best_values, best_cost))


        # Sort by cost and construct RCL
        costs.sort(key=lambda x: x[2])
        RCL = [c for c in costs if c[2] <= costs[0][2] + options.alpha * (costs[-1][2] - costs[0][2])]

        # Choose randomly from the RCL
        selected = random.choice(len(RCL))
        group, best_values, _ = RCL[selected]
        for (i, j), value in best_values.items():
            P[i, j] = value

        # Check for repetition and mutate if necessary
        current_pattern_tuple = pattern_to_tuple(P)
        while current_pattern_tuple in tested_patterns and repetition_count < options.max_repetitions:            
            P = perturb_pattern(P, local_perturbation_rate)
            current_pattern_tuple = pattern_to_tuple(P)
            repetition_count += 1

            if repetition_count >= options.max_repetitions:
                stagnation_count += 1
                break

        # Update cost and stagnation
        current_cost = costs[0][2]
        if abs(last_cost - current_cost) < options.tolerance:
            stagnation_count += 1
            if stagnation_count >= options.max_stagnation:
                local_perturbation_rate = max(0.01, local_perturbation_rate * 0.9)  # Reduce by 10%
                num_perturb = max(1, int(local_perturbation_rate * n * n))
                perturb_indices = random.choice(n * n, num_perturb, replace=False)
                for index in perturb_indices:
                    x, y = divmod(index, n)
                    P[x, y] = random.choice([1, -1])
                stagnation_count = 0
        else:
            stagnation_count = 0

        # Update candidate list (remove the selected group)
        candidate_list = [pos for pos in candidate_list if pos not in group]
        last_cost = current_cost

    return P

def generate_complete_neighborhood(P: ndarray, neighborhood_size_factor: float, mutation_factor: float, tested_patterns: set[tuple]) -> list[ndarray]:
    """
    Generate the complete neighborhood of a pattern {P} by changing {mutation_count} elements.
    Avoids patterns already tested in the tabu list.

    :param P: Initial pattern
    :type P: np.ndarray

    :param neighborhood_size_factor: Size of the neighborhood as a fraction of the pattern size
    :type neighborhood_size: float
    
    :param mutation_factor: Fraction of elements to mutate in the neighborhood
    :type mutation_factor: float
    
    :param tested_patterns: Set of tested patterns
    :type tested_patterns: set[tuple]
    
    :return: List of neighbors
    :rtype: list[np.ndarray]
    """
    
    # Initialize the neighborhood
    n, m = P.shape
    neighborhood = []
    tested_neighbors = set()
    iteration_count = 0
    
    neighborhood_size = max(1, int(neighborhood_size_factor * n * m))
    mutation_count = max(1, int(mutation_factor * n * m))
    all_indices = arange(n * m)
    
    # Generate neighbors until the neighborhood is full
    while len(neighborhood) < neighborhood_size and iteration_count < n * m:
        mutation_indices = random.choice(all_indices, mutation_count, replace=False)
        
        # Mutate the neighbor by flipping the values at the mutation indices
        neighbor = copy(P)
        put(neighbor, mutation_indices, -take(neighbor, mutation_indices))  # Efficient mutation

        
        # Add the neighbor to the neighborhood if it is unique
        neighbor_tuple = pattern_to_tuple(neighbor)
        if neighbor_tuple not in tested_patterns and neighbor_tuple not in tested_neighbors:
            neighborhood.append(neighbor)
            tested_neighbors.add(neighbor_tuple)
    
    return neighborhood

def optimized_local_search(M: ndarray, P: ndarray, tested_patterns: set[tuple], options: LocalSearchOptions) -> ndarray:
    """
    Optimized local search algorithm with tabu list management.
    
    Local search is an iterative method that explores the neighborhood
    of an initial pattern P to find an optimal pattern. Patterns
    already tested are added to the tabu list to avoid redundancies.
    
    :param M: Target matrix
    :type M: np.ndarray
    
    :param P: Initial pattern
    :type P: np.ndarray
    
    :param tested_patterns: Set of tested patterns
    :type tested_patterns: set[tuple]
    
    :param options: Local search options
    :type options: LocalSearchOptions
    
    :return: Optimal pattern
    :rtype: np.ndarray
    """
    
    # Initialize the current pattern
    current_pattern = copy(P)
    
    # Initialize variables
    improved = True
    stagnation_count = 0
    max_stagnation = options.max_stagnation
    
    # While there is improvement
    while improved and stagnation_count < max_stagnation:
        improved = False
        stagnation_count += 1
        
        # Generate the neighborhood based on k (excluding tabu patterns)
        neighborhood = generate_complete_neighborhood(current_pattern, options.max_neighborhood_size, options.neighborhood_mutation_factor, tested_patterns)
        
        # Break if the neighborhood is empty
        if not neighborhood:
            break
        
        # Compare neighbors to find the best one
        best_neighbor = None
        for neighbor in neighborhood:
            # Add to tested patterns
            tested_patterns.add(pattern_to_tuple(neighbor))
        
            if compareP1betterthanP2(M, neighbor, current_pattern):
                best_neighbor = neighbor
                improved = True
        
        if improved:
            current_pattern = best_neighbor