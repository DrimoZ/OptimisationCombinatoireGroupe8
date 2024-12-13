from numpy import ndarray, ones, random, inf, array, meshgrid
from typing import Set
from itertools import combinations, product
from copy import deepcopy
from random import sample, choices

from utils.problem_objective import fobj
from options import GraspConstructionOptions


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

def grasp_constructive_optimized(M: ndarray, tested_patterns: Set[tuple], options: GraspConstructionOptions) -> ndarray:
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
    n = M.shape[0]
    P = random.choice([1, -1], size=(n, n))
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
                P_temp = deepcopy(P)
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
            
            print("Repetition detected, perturbing pattern...")
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