import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from bbb.parameters import Parameters


class CNN(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()

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
            b_x = Variable(inputs, requires_grad=False)
            b_y = Variable(labels)
        
            output = self(b_x)
            loss = self.criterion(output, b_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss

    def eval(self, test_data):
        self.model.eval()

        for inputs, labels in test_data:
            test_output = self(inputs)
            pred_y = torch.max(test_output, 1).indices
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))

        return accuracy
