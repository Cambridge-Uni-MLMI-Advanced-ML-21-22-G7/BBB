import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from bbb.config.parameters import Parameters

class BaseModel(torch.nn.Module):
    def __init__(self, params: Parameters) -> None:
        """Base model class from which all other models are derived.
        
        Methods and parameters common to all model types should be
        set here.

        :param params: model parameters
        :type params: Parameters
        """
        super().__init__()

        self.name = params.name

        # Determine when this run has begun
        self.init_time_str =  datetime.utcnow().strftime("%Y-%m-%d-%H.%M.%S")

        # Create a timestamped directory for the model results
        self.model_save_dir = os.path.join(params.model_save_basedir, params.name, self.init_time_str)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Details for saving the model during training
        self.save_model_path = os.path.join(self.model_save_dir, f'model.pt')

        # Details for saving model training history
        self.save_loss_path = os.path.join(self.model_save_dir, f'loss.npy')
        self.save_eval_metric_path = os.path.join(self.model_save_dir, f'eval_metric.npy')
        self.save_plot_path = os.path.join(self.model_save_dir, f'plot.png')

        # Details for saving metrics to tensorboard during training
        self.save_tensorboard_path = os.path.join(params.tensorboard_save_dir, f'{params.name}_{self.init_time_str}.pt')
        if not os.path.exists(params.tensorboard_save_dir):
            os.makedirs(params.tensorboard_save_dir)
        self.writer = SummaryWriter(self.save_tensorboard_path)

        # Early stopping criteria
        self.early_stopping = params.early_stopping
        self.early_stopping_thresh = params.early_stopping_thresh

        # Arrays to maintain history
        self.loss_hist = []
        self.eval_metric_hist = []

    def load_saved(self, model_path: str):
        """Load a saved model.

        :raises FileNotFoundError: expected saved model file does not exist
        """
        # Load the model from the passed path
        self.save_model_path = model_path

        if os.path.isfile(self.save_model_path):
            self.model.load_state_dict(torch.load(self.save_model_path))
        else:
            raise FileNotFoundError(
                f'No model saved at: {self.save_model_path}'
            )
