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
    alpha: float = 0.9999
    lambda_: float = 0.9
    tolerance: float = 0.001
    max_repetitions: int = 50
    max_stagnation: int = 500
    perturbation_rate: float = 0.70

@dataclass
class GraspOptions:
    iterations: int = 500
    
    construction_options: GraspConstructionOptions = field(default_factory=GraspConstructionOptions)
    vns_options: VnsOptions = field(default_factory=VnsOptions)