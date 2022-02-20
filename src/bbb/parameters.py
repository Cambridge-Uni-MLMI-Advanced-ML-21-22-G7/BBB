from typing import List, Union
from dataclasses import dataclass

from bbb.constants import MODEL_SAVE_DIR


@dataclass
class PriorParameters:
    w_sigma: Union[int, float]
    b_sigma: Union[int, float]

@dataclass
class Parameters:
    name: str
    input_dim: int
    output_dim: int
    hidden_units: int
    weight_mu: List[float]                      # range for mu 
    weight_rho: List[float]                     # range for rho
    prior_params: PriorParameters
    batch_size: int
    lr: float
    epochs: int
    elbo_samples: int                           # to draw for ELBO (training)
    inference_samples: int                      # to draw for inference
    save_dir: str = MODEL_SAVE_DIR


