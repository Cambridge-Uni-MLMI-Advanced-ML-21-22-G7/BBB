import logging
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.constants import KL_REWEIGHTING_TYPES
from bbb.config.parameters import Parameters
from bbb.models.base import BaseModel
from bbb.models.layers import BFC
from bbb.models.evaluation import RegressionEval, ClassificationEval


logger = logging.getLogger(__name__)


class BaseBNN(BaseModel, ABC):
    """Bayesian (Weights) Neural Network
    
    This class inherits from BaseModel, and is inherited by specific
    Regression and Classification classes. See below.
    """
    def __init__(self, params: Parameters) -> None:
        super().__init__(params=params)

        # Parameters
        self.input_dim = params.input_dim
        self.hidden_units = params.hidden_units
        self.hidden_layers = params.hidden_layers
        self.output_dim = params.output_dim
        self.weight_mu = params.weight_mu
        self.weight_rho = params.weight_rho
        self.prior_params = params.prior_params
        self.elbo_samples = params.elbo_samples
        self.inference_samples = params.inference_samples
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.kl_reweighting_type = params.kl_reweighting_type
        self.vp_variance_type = params.vp_variance_type

        # BFC argument dict
        bfc_arguments = {
            "weight_mu": self.weight_mu,
            "weight_rho": self.weight_rho,
            "prior_params": self.prior_params,
            "vp_var_type": self.vp_variance_type
        }

        # Model
        model_layers = []
        model_layers.append(BFC(
            dim_in=self.input_dim,
            dim_out=self.hidden_units,
            **bfc_arguments)
        )
        model_layers.append(nn.ReLU())
        for _ in range(self.hidden_layers-1):
            model_layers.append(BFC(
                dim_in=self.hidden_units,
                dim_out=self.hidden_units,
                **bfc_arguments
            ))
            model_layers.append(nn.ReLU())
        model_layers.append(BFC(
            dim_in=self.hidden_units,
            dim_out=self.output_dim,
            **bfc_arguments)
        )
        self.model = nn.Sequential(*model_layers)

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

    def forward(self, X: Tensor) -> Tensor:
        """Run the passed data forward through the model.

        :param X: data to run through the model
        :type X: Tensor
        :return: model output
        :rtype: Tensor
        """
        return self.model.forward(X) 

    def inference(self, X: Tensor) -> Tensor:
        """Here we do not draw weights but take the mean.
        Hence we are manually going through the layers.

        :param X: data to run through the model
        :type X: Tensor
        :return: model output
        :rtype: Tensor
        """
        for layer in self.model:
            if layer == BFC:
                X = layer.forward(X, sample=False)
            else:
                X = layer.forward(X)
        return X

    def log_prior(self) -> float:
        """Calculate the log prior; log P(w).

        :return: the log prior
        :rtype: float
        """
        log_prior = 0
        for layer in self.model:
            if type(layer) == BFC:
                log_prior += layer.log_prior

        return log_prior

    def log_var_posterior(self) -> float:
        """Calculate the log variational posterior; log q(w|theta).

        :return: the log prior
        :rtype: float
        """
        log_posterior = 0
        for layer in self.model:
            if type(layer) == BFC:
                log_posterior += layer.log_var_post

        return log_posterior

    @abstractmethod
    def get_nll(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate the negative log likelihood; log P(D|w). This is task
        dependant.

        This is an abstract method that should be implemented in classes
        inheriting this class.

        :param outputs: outputs from the model
        :type outputs: torch.Tensor
        :param targets: targets/true values
        :type targets: torch.Tensor
        :return: the negative log likelihood
        :rtype: float
        """
        raise NotImplementedError()

    def sample_ELBO(self, X: Tensor, Y: Tensor, pi: float, num_samples: int) -> Tuple[float, float, float, float]:
        """Run X through the (sampled) model <num_samples> times.
        
        pi is the KL re-weighting factor used in section 3.4.

        :param X: features
        :type X: Tensor
        :param Y: ground-truth/labels/targets
        :type Y: Tensor
        :param pi: weighting to use in KL reweighting
        :type pi: float
        :param num_samples: number of samples to use to estimate expectation
        :type num_samples: int
        :return: ELBO, log prior, log variational posterior and negative log likelihood
        :rtype: Tuple[float, float, float, float]
        """
        # Initialise vectors to hold the components necessary to determin the ELBO
        log_priors = torch.zeros(num_samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(num_samples).to(DEVICE)
        nlls = torch.zeros(num_samples).to(DEVICE)

        # Generate num_samples values.
        for i in range(num_samples):
            preds = self.forward(X)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_var_posterior()
            nlls[i] = self.get_nll(preds, Y)

        # Take the mean of the probabilities; expectation via samples
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        nll = nlls.mean()

        # Compute an estimate of ELBO
        # section 3.4 for pi description
        # pi should not be applied to the NLL
        elbo = pi*(log_variational_posterior - log_prior) + nll

        return elbo, log_prior, log_variational_posterior, nll


    def train(self, train_data: DataLoader) -> float:
        """Single epoch of training.

        :param train_data: training data
        :type train_data: DataLoader
        :raises RuntimeError: unknown KL reweighting type specified
        :return: ELBO of final batch of training data processed
        :rtype: float
        """
        # Put model in training mode
        self.model.train()

        # Loop through the training data
        for idx, (X, Y) in enumerate(train_data):
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            # Calculate pi according to the chosen method
            # Note that the method presented in the paper requires idx
            if self.kl_reweighting_type == KL_REWEIGHTING_TYPES.simple:
                pi = 1/len(train_data)
            elif self.kl_reweighting_type == KL_REWEIGHTING_TYPES.paper:
                pi = 2 ** (len(train_data) - (idx + 1)) / (2 ** len(train_data) - 1)
            else:
                raise RuntimeError(f'Unrecognised KL re-weighting type: {self.kl_reweighting_type}')

            self.zero_grad()

            (
                batch_elbo, batch_log_prior, batch_log_var_post, batch_nll
            ) = self.sample_ELBO(X, Y, pi, self.elbo_samples)
            
            batch_elbo.backward()
            self.optimizer.step()

        # Return the ELBO figure of the final batch as a representative example
        return batch_elbo.item()

    @abstractmethod
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Abstract method: prediction depends on the task.

        :param X: data to run prediction against
        :type X: _type_
        """
        raise NotImplementedError()

    @abstractmethod
    def eval(self, test_data: DataLoader):
        """Abstract method: evaluation depends on the task.

        :param test_data: data to run evaluation against
        :type test_data: DataLoader
        """
        raise NotImplementedError()


class RegressionBNN(RegressionEval, BaseBNN):
    # NOTE: This class inherits from RegressionEval and then BaseBNN
    # The order here is important

    def get_nll(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculation of NLL assuming noise with zero mean and unit variance.
        
        TODO: confirm we want this.
        """
        return -torch.distributions.Normal(outputs, 1.0).log_prob(targets).sum()

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        # Put model into evaluation mode
        self.model.eval()

        # Initialise tensor to hold predictions
        output = torch.zeros(size=[len(X), self.output_dim, self.inference_samples]).to(DEVICE)

        # Repeat forward (sampling) <inference_samples> times
        for i in torch.arange(self.inference_samples):
            output[:,:,i] = self.forward(X)
        
        # Determine the average and the variance of the samples
        mean, var = output.mean(dim=-1), output.var(dim=-1)

        return mean, var

class ClassificationBNN(ClassificationEval, BaseBNN):
    # NOTE: This class inherits from ClassificationEval and then BaseBNN
    # The order here is important

    def forward(self, X: Tensor) -> Tensor:
        # Flatten the image
        X = X.view(-1, self.input_dim)
        return super().forward(X)

    def inference(self, X: Tensor) -> Tensor:
        # Flatten the image
        x = x.view(-1, self.input_dim)
        return super().inference(X)

    def get_nll(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        # NLL calculated as cross entropy
        return F.cross_entropy(outputs, targets, reduction='sum')

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        # Put model into evaluation mode
        self.model.eval()

        # Initialise tensor to hold class probabilities
        probs = torch.zeros(size=[len(X), self.output_dim])

        # Repeat forward (sampling) <inference_samples> times to create probability distribution
        for _ in torch.arange(self.inference_samples):
            output = self.forward(X)

            # Apply softmax to outputs
            out = F.softmax(output, dim=1)

            # Incremental update of average
            probs += out / self.inference_samples
        
        # Select most likely class
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
