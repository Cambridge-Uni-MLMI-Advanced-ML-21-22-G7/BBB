import os

import torch
from torch.utils.tensorboard import SummaryWriter

from bbb.config.parameters import Parameters

class BaseModel(torch.nn.Module):
    def __init__(self, params: Parameters) -> None:
        super().__init__()

        self.name = params.name

        # Save Model
        self.save_model_path = os.path.join(params.model_save_dir, f'{params.name}_model.pt')
        if not os.path.exists(params.model_save_dir):
            os.makedirs(params.model_save_dir)

        # Save Tensorboard
        self.save_tensorboard_path = os.path.join(params.tensorboard_save_dir, f'{params.name}_model.pt')
        if not os.path.exists(params.tensorboard_save_dir):
            os.makedirs(params.tensorboard_save_dir)
        self.writer = SummaryWriter(self.save_tensorboard_path)
