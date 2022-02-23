from typing import List, Union
from dataclasses import dataclass

from bbb.config.constants import MODEL_SAVE_DIR


@dataclass
class PriorParameters:
    w_sigma: Union[int, float]
    b_sigma: Union[int, float]

@dataclass
class Parameters:
    name: str
    input_dim: int
    output_dim: int
    batch_size: int
    lr: float
    epochs: int
    hidden_layers: int = 1
    hidden_units: int = None
    weight_mu: List[float] = None         # range for mu 
    weight_rho: List[float] = None        # range for rho
    prior_params: PriorParameters = None
    elbo_samples: int = None              # to draw for ELBO (training)
    inference_samples: int = None         # to draw for inference
    kl_reweighting_type: int = None       # method used for KL reweighting      
    save_dir: str = MODEL_SAVE_DIR
