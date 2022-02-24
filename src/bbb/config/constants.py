from dataclasses import dataclass


# Directories for saving training artifacts
MODEL_SAVE_DIR = './saved_models'
TENSORBOARD_SAVE_DIR = './tensorboard'

@dataclass
class _KlReweightingTypes:
    """Relates to section 3.4 of the paper:
    method for KL re-weighting when using minibatches.
    """
    simple: int = 0
    paper: int = 1
KL_REWEIGHTING_TYPES = _KlReweightingTypes()

@dataclass
class _VariationalPosteriorVarianceTypes:
    """Relates to section 3.2 of the paper which sets
    sigma = log(1+exp(rho)).

    The simple option sets the variance to simply rho.
    """
    simple: int = 0
    paper: int = 1
VP_VARIANCE_TYPES = _KlReweightingTypes()
