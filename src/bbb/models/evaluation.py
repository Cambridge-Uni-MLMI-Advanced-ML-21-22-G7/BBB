import torch

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters


class ClassificationEval:

    def eval(self, test_data):
        self.model.eval()

        for inputs, labels in test_data:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            test_output = self(inputs)
            pred_y = torch.max(test_output, 1).indices
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))

        self.acc = accuracy
        return accuracy
