from dataclasses import dataclass

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
