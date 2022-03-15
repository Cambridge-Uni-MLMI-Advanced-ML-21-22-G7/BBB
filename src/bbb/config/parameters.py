from typing import List, Union
from dataclasses import dataclass

from bbb.config.constants import MODEL_SAVE_DIR, TENSORBOARD_SAVE_DIR


@dataclass
class PriorParameters:
    """Object for storing parameters of priors.
    """
    w_sigma: Union[int, float]
    b_sigma: Union[int, float]
    w_sigma_2: Union[int, float, None] = None
    b_sigma_2: Union[int, float, None] = None
    w_mixture_weight: Union[int, float, None] = None
    b_mixture_weight: Union[int, float, None] = None

@dataclass
class Parameters:
    """Object for storing training parameters.
    """
    #################################
    # Parameters common to all models
    #################################
    name: str
    input_dim: int
    output_dim: int
    batch_size: int
    epochs: int
    hidden_units: int
    hidden_layers: int

    # Optimiser and Scheduler Parameters
    lr: float
    step_size: int          # Scheduler LR step size (every X epochs)
    opt_choice: str = 'Adam'
    gamma: float = 0.1      # Multiplicative LR decay factor

    ##################################
    # Parameters common to some models
    ##################################

    # VI Parameters
    weight_mu_range: List[float] = None        # range for mu 
    weight_rho_range: List[float] = None       # range for rho
    prior_params: PriorParameters = None
    elbo_samples: int = None                   # to draw for ELBO (training)
    inference_samples: int = None              # to draw for inference
    regression_likelihood_noise: float = None  # noise used when determining likelihood of regression problems

    # Paper choices parameters
    kl_reweighting_type: int = None            # method used for KL reweighting
    vp_variance_type: int = None               # type of variational posterior variance used
    prior_type: int = None
    local_reparam_trick: bool = False

    # Dropout paramters
    dropout: bool = False
    dropout_p: float = None

    # Early stopping parameters
    # By default, do not early stop
    early_stopping: bool = False
    early_stopping_thresh: float = 0

    # These parameters are unlikely to need to
    # be changed from their defaults
    model_save_basedir: str = MODEL_SAVE_DIR
    tensorboard_save_dir: str = TENSORBOARD_SAVE_DIR
