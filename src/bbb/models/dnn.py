from typing import Tuple, List
from abc import ABC, abstractmethod

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters
from bbb.models.base import BaseModel
from bbb.models.evaluation import RegressionEval, ClassificationEval


class BaseDNN(BaseModel, ABC):
    def __init__(self, params: Parameters, eval_mode: bool = False) -> None:
        """Vanilla DNN with customisable number of hidden layers.

        This class inherits from BaseModel and then ABC.
        The order here is important.

        :param params: model parameters
        :type params: Parameters
        :param eval_mode: evaluation mode, defaults to False
        :type eval_mode: bool
        """
        super().__init__(params=params, eval_mode=eval_mode)

        ############
        # Parameters
        ############

        self.input_dim = params.input_dim
        self.output_dim = params.output_dim
        self.hidden_layers = params.hidden_layers
        self.hidden_units = params.hidden_units
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.step_size = params.step_size
        self.dropout = params.dropout
        self.dropout_p = params.dropout_p

        #######
        # Model
        #######
        # Initialise empty list of layers
        model_layers = []

        # Add the first layer
        model_layers.append(nn.Linear(
            in_features=self.input_dim,
            out_features=self.hidden_units
        ))
        model_layers.append(nn.ReLU())
        if self.dropout:
            model_layers.append(nn.Dropout(p=self.dropout_p))

        # Add the self.hidden_layers-2 intermediate layers
        for _ in range(self.hidden_layers-2):
            model_layers.append(nn.Linear(
                    in_features=self.hidden_units,
                    out_features=self.hidden_units
                ))
            model_layers.append(nn.ReLU())
            if self.dropout:
                model_layers.append(nn.Dropout(p=self.dropout_p))

        # Add the final layer
        model_layers.append(nn.Linear(
            in_features=self.hidden_units,
            out_features=self.output_dim
        ))

        # Pass the list to the nn.Sequential class
        self.model = nn.Sequential(*model_layers)
        
        ###########
        # Optimizer
        ###########
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr
        )

        ###########
        # Scheduler
        ###########
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=0.5
        )

    def forward(self, X: Tensor):
        return self.model.forward(X)

    def train_step(self, train_data: DataLoader):
        self.train()

        for i, (X, Y) in enumerate(train_data):
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            self.optimizer.zero_grad()

            output = self(X)
            loss = self.criterion(output, Y)
            
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

    @abstractmethod
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Abstract method: prediction depends on the task.

        :param X: data to run prediction against
        :type X: _type_
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, test_data: DataLoader):
        """Abstract method: evaluation depends on the task.

        :param test_data: data to run evaluation against
        :type test_data: DataLoader
        """
        raise NotImplementedError()


class RegressionDNN(RegressionEval, BaseDNN):
    # NOTE: This class inherits from RegressionEval and then BaseDNN
    # The order here is important

    def __init__(self, params: Parameters, eval_mode: bool = False) -> None:
        super().__init__(params=params, eval_mode=eval_mode)
        
        # Criterion
        self.criterion = nn.MSELoss()

    def predict(self, X: Tensor) -> Tuple[Tensor, None]:
        # Ensure tensor is assigned to correct device
        X = X.to(DEVICE)

        # Put model in evaluation mode
        self.eval()

        # Make predictions
        preds = self.forward(X)

        # Return tuple with preds and None
        # BBB methods return mean and variance of samples NNs
        # Hence, using None here to keep return dimensions consistent
        return preds, None, None

class ClassificationDNN(ClassificationEval, BaseDNN):
    # NOTE: This class inherits from ClassificationEval and then BaseDNN
    # The order here is important

    def __init__(self, params: Parameters, eval_mode: bool = False) -> None:
        super().__init__(params=params, eval_mode=eval_mode)
        
        # Criterion
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X: Tensor) -> Tensor:
        X = X.view(-1, self.input_dim)
        return F.softmax(super().forward(X), dim=1)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        # Ensure tensor is assigned to correct device
        X = X.to(DEVICE)

        # Put model into evaluation mode
        self.eval()

        # Pass the input through the model
        probs = self.forward(X)
        
        # Select most likely class
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
