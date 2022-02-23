import torch
import torch.nn.functional as F

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters


class RegressionEval:

    eval_metric = 'RMSE'

    def eval(self, test_data):
        self.model.eval()
        running_err = 0
        total = 0

        with torch.no_grad():
            for X, Y in test_data:
                X = X.to(DEVICE)
                Y = Y.to(DEVICE)

                pred_Y = self(X)

                total += self.batch_size
                running_err += ((pred_Y - Y)*(pred_Y - Y)).sum().data

        self.eval_score = torch.sqrt(running_err/total)

class ClassificationEval:

    eval_metric = 'Acc'

    def eval(self, test_data):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_data:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                preds = self(inputs)

                total += self.batch_size
                correct += (labels == preds).sum().item()

        self.eval_score = correct / total
