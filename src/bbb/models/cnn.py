from torch import nn, optim

from bbb.config.parameters import Parameters
from bbb.models.base import BaseModel
from bbb.models.evaluation import ClassificationEval

class CNN(ClassificationEval, BaseModel):
    def __init__(self, params: Parameters):
        super().__init__(params=params)

        # Parameters
        self.input_dim = params.input_dim # params.get('input_dim', "default value")
        self.output_dim = params.output_dim
        self.batch_size = params.batch_size
        self.lr = params.lr

        # Model
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=16*12*12,
                out_features=self.output_dim
            ),
            # nn.Softmax()
        )

        # Criterion
        self.criterion = nn.CrossEntropyLoss()
    
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    def forward(self, x):
        return self.model.forward(x)

    def train(self, train_data):
        self.model.train()

        for i, (inputs, labels) in enumerate(train_data):
            output = self(inputs)
            loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss
