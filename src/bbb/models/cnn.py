from typing import Union, List

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters
from bbb.models.base import BaseModel
from bbb.models.evaluation import ClassificationEval

class CNN(ClassificationEval, BaseModel):
    def __init__(self, params: Parameters, eval_mode: bool = False) -> None:
        """Vanilla CNN.

        This class inherits from ClassificationEval and then BaseModel.
        The order here is important.

        :param params: model parameters
        :type params: Parameters
        """
        super().__init__(params=params, eval_mode=eval_mode)

        # Parameters
        self.input_dim = params.input_dim # params.get('input_dim', "default value")
        self.output_dim = params.output_dim
        self.hidden_layers = params.hidden_layers
        self.hidden_units = params.hidden_units
        self.batch_size = params.batch_size
        self.lr = params.lr

        # Model
        model_layers = []
        model_layers.append(nn.Linear(
            in_features=self.input_dim,
            out_features=self.hidden_units
        ))
        model_layers.append(nn.ReLU())
        for _ in range(self.hidden_layers-2):
            model_layers.append(nn.Linear(
                    in_features=self.hidden_units,
                    out_features=self.hidden_units
                ))
            model_layers.append(nn.ReLU())
        model_layers.append(nn.Linear(
            in_features=self.hidden_units,
            out_features=self.output_dim
        ))
        model_layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*model_layers)

        # Criterion
        self.criterion = nn.CrossEntropyLoss()
    
        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.5
        )

    def forward(self, X: Tensor) -> Tensor:
        X = X.view(-1, self.input_dim)
        return self.model.forward(X)

    def train_step(self, train_data: DataLoader) -> float:
        # Put model into training mode
        self.train()

        # Loop through the training data
        for _, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()

            output = self(inputs)
            loss = self.criterion(output, labels)
            
            loss.backward()
            self.optimizer.step()

        # Record the loss
        self.loss_hist.append(loss.item())
            
        return loss

    def weight_samples(self) -> List[Tensor]:
        """Sample the BFC layer weights.

        :return: weight samples
        :rtype: List[Tensor]
        """
        return [param.flatten() for name, param in self.model.named_parameters() if name.endswith('weight')]

    def predict(self, X: Tensor) -> Union[Tensor, Tensor]:
        # Put model into evaluation mode
        self.eval()

        # Pass the input through the model
        output = self.forward(X)

        # Apply softmax to the output to get probs
        probs = F.softmax(output, dim=1)
        
        # Select most likely class
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
