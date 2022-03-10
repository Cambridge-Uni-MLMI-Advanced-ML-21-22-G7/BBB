from dataclasses import dataclass


# Directories for saving training artifacts
MODEL_SAVE_DIR = './saved_models'
TENSORBOARD_SAVE_DIR = './tensorboard'
MUSHROOM_DATASET_PATH = '/homes/yw575/AML/BBB/data/mushroom.csv'

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
VP_VARIANCE_TYPES = _VariationalPosteriorVarianceTypes()

@dataclass
class _PriorTypes:
    """Relates to section 3.3 of the paper which uses
    a scale mixture prior.

    The single option uses a singe Gaussian.
    """
    single: int = 0
    mixture: int = 1
    laplacian: int = 2
PRIOR_TYPES = _PriorTypes()
