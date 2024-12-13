from dataclasses import dataclass, field

@dataclass
class MatrixOptions:
    n: int = 15

@dataclass
class VnsOptions:
    k_max: int = 1
    max_duration: int = 4 * 1_000_000
    
@dataclass
class GraspConstructionOptions:
    alpha: float = 0.99
    lambda_: float = 0.01
    tolerance: float = 0.1
    max_repetitions: int = 50
    max_stagnation: int = 10
    perturbation_rate: float = 0.3
    max_fixed_candidates_per_iteration: int = 5

@dataclass
class GraspOptions:
    iterations: int = 10000
    
    construction_options: GraspConstructionOptions = field(default_factory=GraspConstructionOptions)
    vns_options: VnsOptions = field(default_factory=VnsOptions)