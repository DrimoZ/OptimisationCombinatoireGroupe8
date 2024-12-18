from dataclasses import dataclass, field

@dataclass
class MatrixOptions:
    n: int = 40

@dataclass
class VnsOptions:
    k_max: int = 1
    max_duration: int = 4 * 1_000_000
    
@dataclass
class LocalSearchOptions:
    max_neighborhood_size : float = 0.1
    neighborhood_mutation_factor: float = 0.05
    max_stagnation: int = 5
    
@dataclass
class GraspConstructionOptions:
    alpha: float = 0.6
    lambda_: float = 0.3
    tolerance: float = 0.0001
    max_repetitions: int = 50
    max_stagnation: int = 10
    perturbation_rate: float = 0.3
    max_fixed_candidates_per_iteration: int = 3

@dataclass
class GraspOptions:
    iterations: int = 5000
    tested_list_max_size: int = 10_000
    
    construction_options: GraspConstructionOptions = field(default_factory=GraspConstructionOptions)
    local_search_options: LocalSearchOptions = field(default_factory=LocalSearchOptions)
    vns_options: VnsOptions = field(default_factory=VnsOptions)