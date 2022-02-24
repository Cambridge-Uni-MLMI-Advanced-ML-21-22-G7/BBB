import os
import logging
from abc import ABC, abstractmethod

import torch
from torch import nn, optim
import torch.nn.functional as F

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.constants import KL_REWEIGHTING_TYPES
from bbb.config.parameters import Parameters
from bbb.models.base import BaseModel
from bbb.models.layers import BFC
from bbb.models.evaluation import RegressionEval, ClassificationEval


logger = logging.getLogger(__name__)


class BaseBNN(BaseModel, ABC):
    """ Bayesian (Weights) Neural Network """
    def __init__(self, params: Parameters) -> None:
        super().__init__(params=params)

        # Parameters
        self.input_dim = params.input_dim
        self.hidden_units = params.hidden_units
        self.output_dim = params.output_dim
        self.weight_mu = params.weight_mu
        self.weight_rho = params.weight_rho
        self.prior_params = params.prior_params
        self.elbo_samples = params.elbo_samples # num samples to draw for ELBO
        self.inference_samples = params.inference_samples # num samples to draw for ELBO
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.kl_reweighting_type = params.kl_reweighting_type

        # Model
        self.model = nn.Sequential(
            BFC(self.input_dim, self.hidden_units, self.weight_mu, self.weight_rho, self.prior_params), 
            nn.ReLU(),
            BFC(self.hidden_units, self.hidden_units, self.weight_mu, self.weight_rho, self.prior_params),
            nn.ReLU(),
            BFC(self.hidden_units, self.output_dim, self.weight_mu, self.weight_rho, self.prior_params)
        )

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

    def inference(self, x):
        """Here we do not draw weights but take the mean.
        Hence we are manually going through the layers.
        """
        for layer in self.model:
            if layer == BFC:
                x = layer.forward(x, sample=False)
            else:
                x = layer.forward(x)
        return x

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
        """Calculate the negative log likelihood; log P(D|w).

        :param outputs: outputs from the model
        :type outputs: torch.Tensor
        :param targets: targets/true values
        :type targets: torch.Tensor
        :return: the negative log likelihood
        :rtype: float
        """
        raise NotImplementedError()

    def sample_ELBO(self, x, y, pi, num_samples):
        """Run X through the (sampled) model <samples> times.
        
        pi is the KL re-weighting factor used in section 3.4.
        """
        log_priors = torch.zeros(num_samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(num_samples).to(DEVICE)
        nlls = torch.zeros(num_samples).to(DEVICE)

        for i in range(num_samples):
            preds = self.forward(x)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_var_posterior()
            nlls[i] = self.get_nll(preds, y)

        # Compute an estimate of ELBO
        log_prior = pi*log_priors.mean()  # section 3.4 for pi description
        log_variational_posterior = pi*log_variational_posteriors.mean()
        nll = nlls.mean()  # pi should not be applied to the NLL

        elbo = log_variational_posterior - log_prior + nll
        return elbo, log_priors.mean(), log_variational_posteriors.mean(), nll


    def train(self, train_data) -> float:
        self.model.train()

        for idx, (X, Y) in enumerate(train_data):
            if self.kl_reweighting_type == KL_REWEIGHTING_TYPES.simple:
                pi = 1/len(train_data)
            elif self.kl_reweighting_type == KL_REWEIGHTING_TYPES.paper:
                pi = 2 ** (len(train_data) - (idx + 1)) / (2 ** len(train_data) - 1)
            else:
                raise RuntimeError(f'Unrecognised KL re-weighting type: {self.kl_reweighting_type}')

            X, Y = X.to(DEVICE), Y.to(DEVICE)
            self.zero_grad()
            (
                batch_elbo, batch_log_prior, batch_log_var_post, batch_nll
            ) = self.sample_ELBO(X, Y, pi, self.elbo_samples)
            
            batch_elbo.backward()
            self.optimizer.step()

        # logger.debug(f'ELBO: {batch_elbo.item()}')

        return batch_elbo.item()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    @abstractmethod
    def eval(self, test_data):
        raise NotImplementedError()


class RegressionBNN(RegressionEval, BaseBNN):
    def get_nll(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        # TODO: Assuming noise with zero mean and unit variance; confirm we want this
        return -torch.distributions.Normal(outputs, 1.0).log_prob(targets).sum()

    def predict(self, X):
        output = torch.zeros(size=[len(X), self.output_dim, self.inference_samples]).to(DEVICE)

        # Repeat forward (sampling) <inference_samples> times to create probability distrib
        for i in torch.arange(self.inference_samples):
            output[:,:,i] = self.forward(X)
        
        # Determine the average and the variance of the samples
        mean, var = output.mean(dim=-1), output.var(dim=-1)

        return mean, var

class ClassificationBNN(ClassificationEval, BaseBNN):
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return super().forward(x)

    def inference(self, x):
        x = x.view(-1, self.input_dim)
        return super().inference(x)

    def get_nll(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return F.cross_entropy(outputs, targets, reduction='sum')

    def predict(self, X):
        self.model.eval()
        probs = torch.zeros(size=[len(X), self.output_dim])

        # Repeat forward (sampling) <inference_samples> times to create probability distrib
        for _ in torch.arange(self.inference_samples):
            output = self.forward(X)
            out = F.softmax(output, dim=1)
            probs += out / self.inference_samples
        
        # Select most likely class
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
