import torch
from torch.utils.data import DataLoader

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters


class RegressionEval:
    # Evaluation method for regression tasks.
    # Should be the first class inherited by all regression models.

    # This will be shown in the tqdm progress bar
    # and in Tensorboard.
    eval_metric = 'RMSE'

    def evaluate(self, test_data: DataLoader) -> float:
        # Put model in evaluation mode
        self.eval()

        running_err = 0
        total = 0

        with torch.no_grad():
            for X, Y in test_data:
                X = X.to(DEVICE)
                Y = Y.to(DEVICE)

                pred_Y, _, _ = self.predict(X)

                total += self.batch_size
                running_err += ((pred_Y - Y)*(pred_Y - Y)).sum().data

        self.eval_score = torch.sqrt(running_err/total)

        # Record the evaluation metric score
        self.eval_metric_hist.append(self.eval_score.item())

        return self.eval_score

class ClassificationEval:
    # Evaluation method for classification tasks.
    # Should be the first class inherited by all classification models.

    # This will be shown in the tqdm progress bar
    # and in Tensorboard.
    eval_metric = 'Acc'

    def evaluate(self, test_data: DataLoader) -> float:
        # Put model in evaluation mode
        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_data:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                preds, probs = self.predict(inputs)

                total += self.batch_size
                correct += (labels == preds).sum()

        self.eval_score = correct / total

        # Record the evaluation metric score
        self.eval_metric_hist.append(self.eval_score.item())

        return self.eval_score
