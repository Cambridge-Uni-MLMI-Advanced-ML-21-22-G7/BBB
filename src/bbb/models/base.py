import os

import torch

from bbb.config.parameters import Parameters


class BaseModel(torch.nn.Module):
    def __init__(self, params: Parameters) -> None:
        super().__init__()

        self.name = params.name

        # Save Model
        self.save_model_path = os.path.join(params.save_dir, f'{params.name}_model.pt')
        if not os.path.exists(params.save_dir):
            os.makedirs(params.save_dir)
