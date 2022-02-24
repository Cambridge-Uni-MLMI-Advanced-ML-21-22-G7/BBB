from typing import List, Union
from dataclasses import dataclass

from bbb.config.constants import MODEL_SAVE_DIR, TENSORBOARD_SAVE_DIR


@dataclass
class PriorParameters:
    """Object for storing parameters of priors.
    """
    w_sigma: Union[int, float]
    b_sigma: Union[int, float]

@dataclass
class Parameters:
    """Object for storing training parameters.
    """
    # Parameters common to all models
    name: str
    input_dim: int
    output_dim: int
    batch_size: int
    lr: float
    epochs: int
    hidden_layers: int = 1

    # Parameters common to some models
    # All default to None
    hidden_units: int = None
    weight_mu: List[float] = None         # range for mu 
    weight_rho: List[float] = None        # range for rho
    prior_params: PriorParameters = None
    elbo_samples: int = None              # to draw for ELBO (training)
    inference_samples: int = None         # to draw for inference
    kl_reweighting_type: int = None       # method used for KL reweighting
    vp_variance_type: int = None          # type of variational posterior variance used

    # These parameters are unlikely to need to
    # be changed from their defaults
    model_save_dir: str = MODEL_SAVE_DIR
    tensorboard_save_dir: str = TENSORBOARD_SAVE_DIR
