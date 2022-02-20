import torch
from torch import nn, optim
import torch.nn.functional as F

from bbb.parameters import Parameters


class DNN(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()

        # Parameters
        self.input_dim = params.input_dim # params.get('input_dim', "default value")
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
            self.optimizer,step_size=100,
            gamma=0.5
        )


    def forward(self, x):
        return self.model.forward(x)

    def train(self, train_data):
        self.model.train()

        for i, (x, y) in enumerate(train_data):
            output = self(x)
            loss = self.criterion(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss

    def eval(self, test_data):
        self.model.eval()

        n = 1
        avg = 0
        for x, y in test_data:
            pred_y = self(x)
            mse = self.criterion(pred_y, y)

            avg = (1/n)*mse + ((n-1)/n)*avg
            n += 1

        return avg

    def predict(self, predict_data):
        self.model.eval()
        return self(predict_data)
