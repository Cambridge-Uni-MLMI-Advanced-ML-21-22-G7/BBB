from typing import Tuple

from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

from bbb.config.parameters import Parameters
from bbb.models.base import BaseModel
from bbb.models.evaluation import RegressionEval


class DNN(RegressionEval, BaseModel):
    def __init__(self, params: Parameters) -> None:
        """Vanilla DNN with customisable number of hidden layers.

        This class inherits from RegressionEval and then BaseModel.
        The order here is important.

        :param params: model parameters
        :type params: Parameters
        """
        super().__init__(params=params)

        # Parameters
        self.input_dim = params.input_dim
        self.output_dim = params.output_dim
        self.hidden_layers = params.hidden_layers
        self.hidden_units = params.hidden_units
        self.batch_size = params.batch_size
        self.lr = params.lr

        # Model
        self.model = nn.Sequential( 
            nn.Linear(
                in_features=self.input_dim,
                out_features=self.hidden_units
            ),
            nn.ReLU(),
            *[
                nn.Linear(
                    in_features=self.hidden_units,
                    out_features=self.hidden_units
                ),
                nn.ReLU(),
            ]*self.hidden_layers,
            nn.Linear(
                in_features=self.hidden_units,
                out_features=self.output_dim
            ),
        )

        # Criterion
        self.criterion = nn.MSELoss()
    
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.5
        )


    def forward(self, X: Tensor):
        return self.model.forward(X)

    def train(self, train_data: DataLoader):
        self.model.train()

        for i, (x, y) in enumerate(train_data):
            output = self(x)
            loss = self.criterion(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss

    def predict(self, X: Tensor) -> Tuple[Tensor, None]:
        # Put model in evaluation mode
        self.model.eval()

        # Make predictions
        preds = self(X)

        # Return tuple with preds and None
        # BBB methods return mean and variance of samples NNs
        # Hence, using None here to keep return dimensions consistent
        return preds, None
