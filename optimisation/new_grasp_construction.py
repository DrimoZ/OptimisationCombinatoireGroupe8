import numpy as np

from typing import Set

from utils.problem_objective import fobj
from options import GraspConstructionOptions


def pattern_to_tuple(P: np.ndarray) -> tuple:
    """
    Convert a pattern matrix P (np.ndarray) into a tuple of tuples to make it hashable.
    """
    return tuple(tuple(row) for row in P)

def perturb_pattern(P: np.ndarray, perturbation_rate: float) -> np.ndarray:
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
    perturb_indices = np.random.choice(n * n, num_perturb, replace=False)
    
    for index in perturb_indices:
        x, y = divmod(index, n)
        P[x, y] = np.random.choice([1, -1])  # Randomly flip the value at position (x, y)
    
    return P

def grasp_constructive_optimized(M: np.ndarray, tested_patterns: Set[tuple], options: GraspConstructionOptions) -> np.ndarray:
    """
    Optimized GRASP construction algorithm with minimal randomization and perturbation.

    :param M: Target matrix
    :type M: np.ndarray
    
    :param options: GRASP construction options
    :type options: GraspConstructionOptions
    
    :return: Constructed pattern
    :rtype: np.ndarray
    """
    
    n = M.shape[0]
    P = np.random.choice([1, -1], size=(n, n))  # Initialize random pattern
    last_cost = np.inf
    tested_patterns: Set[tuple] = {pattern_to_tuple(P)}  # Store tested patterns
    stagnation_count = 0
    repetition_count = 0

    candidate_list = [(i, j) for i in range(n) for j in range(n)]
    
    perturbation_rate = options.perturbation_rate

    while candidate_list:
        np.random.shuffle(candidate_list)
        costs = []

        # Evaluate all candidates
        for i, j in candidate_list:
            best_cost = np.inf
            best_value = P[i, j]  # Default to current value
            for value in [1, -1]:
                if P[i, j] == value:
                    continue  # Skip redundant checks

                P_temp = P.copy()
                P_temp[i, j] = value
                try:
                    rank, inv_smallest_sv = fobj(M, P_temp)
                    cost = rank + options.lambda_ * inv_smallest_sv
                except IndexError:
                    cost = np.inf

                if cost < best_cost:
                    best_cost = cost
                    best_value = value

            costs.append(((i, j, best_value), best_cost))

        # Sort by cost and select RCL
        costs.sort(key=lambda x: x[1])
        min_cost, max_cost = costs[0][1], costs[-1][1]

        # Simple RCL threshold: Select all candidates with cost <= max_cost
        threshold = max_cost
        RCL = [c for c in costs if c[1] <= threshold]

        # If RCL is empty, print debugging information
        if len(RCL) == 0:
            print(f"RCL is empty. min_cost: {min_cost}, max_cost: {max_cost}")
            RCL = costs  # Fallback to using the entire list if RCL is empty

        # Choose randomly from the RCL
        selected = np.random.choice(len(RCL))
        (i, j, value), _ = RCL[selected]
        P[i, j] = value

        # Check for repetition and mutate if necessary
        current_pattern_tuple = pattern_to_tuple(P)
        while current_pattern_tuple in tested_patterns and repetition_count < options.max_repetitions:
            P = perturb_pattern(P, perturbation_rate)
            current_pattern_tuple = pattern_to_tuple(P)
            repetition_count += 1
            
            if repetition_count >= options.max_repetitions:
                stagnation_count += 1
                break

        # Update cost and stagnation
        current_cost = costs[0][1]
        if np.abs(last_cost - current_cost) < options.tolerance:
            stagnation_count += 1
            if stagnation_count >= options.max_stagnation:
                # Adaptive perturbation: reduce rate over time
                perturbation_rate = max(0.01, perturbation_rate * 0.9)  # Reduce by 10% after stagnation
                num_perturb = max(1, int(perturbation_rate * n * n))
                perturb_indices = np.random.choice(n * n, num_perturb, replace=False)
                for index in perturb_indices:
                    x, y = divmod(index, n)
                    P[x, y] = np.random.choice([1, -1])
                stagnation_count = 0
        else:
            stagnation_count = 0

        # Update candidate list (remove the selected one)
        candidate_list = [(x, y) for x, y in candidate_list if x != i and y != j]
        last_cost = current_cost

    return P