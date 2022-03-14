import logging
from abc import ABC, abstractmethod
from time import sleep
from typing import Tuple, List

import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.constants import KL_REWEIGHTING_TYPES, PRIOR_TYPES
from bbb.config.parameters import Parameters
from bbb.models.base import BaseModel
from bbb.models.layers import BFC, BFC_LRT
from bbb.models.evaluation import RegressionEval, ClassificationEval


logger = logging.getLogger(__name__)


class BaseBNN(BaseModel, ABC):
    """Bayesian (Weights) Neural Network
    
    This class inherits from BaseModel, and is inherited by specific
    Regression and Classification classes. See below.
    """
    def __init__(self, params: Parameters, eval_mode: bool = False) -> None:
        super().__init__(params=params, eval_mode=eval_mode)

        # Parameters
        self.input_dim = params.input_dim
        self.hidden_units = params.hidden_units
        self.hidden_layers = params.hidden_layers
        self.output_dim = params.output_dim
        self.weight_mu_range = params.weight_mu_range
        self.weight_rho_range = params.weight_rho_range
        self.prior_params = params.prior_params
        self.elbo_samples = params.elbo_samples
        self.inference_samples = params.inference_samples
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.step_size = params.step_size
        self.prior_type = params.prior_type
        self.kl_reweighting_type = params.kl_reweighting_type
        self.vp_variance_type = params.vp_variance_type
        self.local_reparam_trick = params.local_reparam_trick
        self.gamma = params.gamma

        # If using local reparameterisation trick the prior must be Gaussian
        # This is due to the exact calculation of the KL divergence
        # KL(q(w|thetat)||p(w))
        if self.local_reparam_trick:
            assert self.prior_type == PRIOR_TYPES.single


        # BFC argument dict
        bfc_arguments = {
            "weight_mu_range": self.weight_mu_range,
            "weight_rho_range": self.weight_rho_range,
            "prior_params": self.prior_params,
            "prior_type": self.prior_type,
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
        for _ in range(self.hidden_layers-2):
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
            self.parameters(),
            lr=self.lr
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma
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
            if isinstance(layer, BFC):
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
            if isinstance(layer, BFC):
                log_prior += layer.log_prior

        return log_prior

    def log_var_posterior(self) -> float:
        """Calculate the log variational posterior; log q(w|theta).

        :return: the log prior
        :rtype: float
        """
        log_posterior = 0
        for layer in self.model:
            if isinstance(layer, BFC):
                log_posterior += layer.log_var_post

        return log_posterior

    def kl_d(self) -> float:
        """Calculate the KL divergence: KL(q(w|theta)||p(w)).
        This is the divergence between the prior and variational posterior.

        :return: the KL divergence
        :rtype: float
        """
        # This should only be run for the local reparameterisation trick
        assert self.local_reparam_trick

        kl_d = 0
        for layer in self.model:
            if isinstance(layer, BFC_LRT):
                kl_d += layer.kl_d

        return kl_d

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

    def sample_ELBO_lrt(self, X: Tensor, Y: Tensor, pi: float, num_samples: int) -> float:
        """Run X through the (sampled) model <num_samples> times. This uses the local
        reparameterisation trick.
        
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
        # Ensure that this method is not being accidentally called from the wrong place
        if not self.local_reparam_trick:
            raise RuntimeError("Free energy calculation applies to local reparameterisation trick only")
        
        # Determine the mean negative log-likelihood over a range of activation samples
        nlls = torch.zeros(num_samples).to(DEVICE)
        for i in range(num_samples):
            preds = self.forward(X)
            nlls[i] = self.get_nll(preds, Y)
        nll = nlls.mean()
        
        # The KL divergence is constant
        kl_d = self.kl_d()

        # Compute an estimate of ELBO
        # section 3.4 for pi description
        # pi should not be applied to the NLL
        elbo = pi*kl_d + nll

        return elbo

    def train_step(self, train_data: DataLoader) -> float:
        """Single epoch of training.

        :param train_data: training data
        :type train_data: DataLoader
        :raises RuntimeError: unknown KL reweighting type specified
        :return: ELBO of final batch of training data processed
        :rtype: float
        """
        # Put model in training mode
        self.train()

        # Determine the number of batches
        num_batches = len(train_data)

        # Loop through the training data
        for idx, (X, Y) in enumerate(train_data):
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            # Calculate pi according to the chosen method
            # Note that the method presented in the paper requires idx
            if self.kl_reweighting_type == KL_REWEIGHTING_TYPES.simple:
                pi = 1/num_batches
            elif self.kl_reweighting_type == KL_REWEIGHTING_TYPES.paper:
                pi = 2 ** (num_batches - (idx + 1)) / (2 ** num_batches - 1)
            else:
                raise RuntimeError(f'Unrecognised KL re-weighting type: {self.kl_reweighting_type}')

            self.zero_grad()

            # Call the appropriate method for determining the sample ELBO
            if self.local_reparam_trick:
                batch_elbo = self.sample_ELBO_lrt(X, Y, pi, self.elbo_samples)
            else:
                (
                    batch_elbo, batch_log_prior, batch_log_var_post, batch_nll
                ) = self.sample_ELBO(X, Y, pi, self.elbo_samples)

            batch_elbo.backward()
            self.optimizer.step()

        # Record the ELBO
        self.loss_hist.append(batch_elbo.item())

        # Return the ELBO figure of the final batch as a representative example
        return batch_elbo.item()

    def weight_samples(self) -> List[Tensor]:
        """Sample the BFC layer weights.

        :return: weight samples
        :rtype: List[Tensor]
        """
        # Put model into evaluation mode
        self.eval()

        ##################################
        # Sampling weights and taking mean
        ##################################
        # Initialise list of tensors to hold weight samples
        # Each tensor will hold samples of weights from a single layer
        weight_tensors = [
            torch.zeros(size=(self.input_dim*self.hidden_units, self.inference_samples)).to(DEVICE)]
        weight_tensors.extend([
            torch.zeros(size=(self.hidden_units**2, self.inference_samples)).to(DEVICE)
            for _ in range(self.hidden_layers-2)
        ])
        weight_tensors.append(torch.zeros(size=(self.hidden_units*self.output_dim, self.inference_samples)).to(DEVICE))

        # Repeat sampling <inference_samples> times
        for i, layer in enumerate([l for l in self.model if isinstance(l, BFC)]):
            for j in range(self.inference_samples):
                weight_tensors[i][:, j] = layer.w_var_post.sample().flatten()
            
             # Take the mean across the samples
            weight_tensors[i] = weight_tensors[i].mean(axis=-1)

        ####################
        # Using mean weights
        ####################
        # weight_tensors = []
        # for layer in [l for l in self.model if isinstance(l, BFC)]:
        #     weight_tensors.append(layer.w_var_post.mu.flatten())

        return weight_tensors

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


class RegressionBNN(RegressionEval, BaseBNN):
    # NOTE: This class inherits from RegressionEval and then BaseBNN
    # The order here is important

    def __init__(self, params: Parameters, eval_mode: bool = False) -> None:
        super().__init__(params=params, eval_mode=eval_mode)
        
        # Assert that regression_likelihood_noise has been provided
        assert type(params.regression_likelihood_noise) == float

        self.regression_likelihood_noise = params.regression_likelihood_noise

    def get_nll(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculation of NLL assuming noise with zero mean and unit variance.
        
        TODO: confirm we want this.
        """
        return -torch.distributions.Normal(outputs, self.regression_likelihood_noise).log_prob(targets).sum()

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        # Ensure tensor is assigned to correct device
        X = X.to(DEVICE)

        # Put model into evaluation mode
        self.eval()

        # Initialise tensor to hold predictions
        output = torch.zeros(size=[len(X), self.output_dim, self.inference_samples]).to(DEVICE)

        # Repeat forward (sampling) <inference_samples> times
        for i in torch.arange(self.inference_samples):
            output[:,:,i] = self.forward(X)
        
        # Determine the average and the variance of the samples
        mean, var = output.mean(dim=-1), output.var(dim=-1)

        # Determine the quartiles
        q = torch.tensor([0., 0.25, 0.75, 1.]).to(DEVICE)
        quartiles = torch.quantile(output, q, dim=-1)

        return mean, var, quartiles

class ClassificationBNN(ClassificationEval, BaseBNN):
    # NOTE: This class inherits from ClassificationEval and then BaseBNN
    # The order here is important

    def forward(self, X: Tensor) -> Tensor:
        # Flatten the image
        X = X.view(-1, self.input_dim)
        return F.softmax(super().forward(X), dim=1)

    def inference(self, X: Tensor) -> Tensor:
        # Flatten the image
        x = x.view(-1, self.input_dim)
        return F.softmax(super().inference(X), dim=1)

    def get_nll(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        # NLL calculated as cross entropy
        return F.cross_entropy(outputs, targets, reduction='sum')

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        # Ensure tensor is assigned to correct device
        X = X.to(DEVICE)

        # Put model into evaluation mode
        self.eval()

        # Initialise tensor to hold class probabilities
        probs = torch.zeros(size=[len(X), self.output_dim]).to(DEVICE)

        # Repeat forward (sampling) <inference_samples> times to create probability distribution
        for _ in torch.arange(self.inference_samples):
            output = self.forward(X)

            # Incremental update of average
            probs += output / self.inference_samples
        
        # Select most likely class
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs

class BanditBNN(RegressionEval, BaseBNN):
    # NOTE: This class inherits from RegressionEval and then BaseBNN
    # The order here is important
    def get_nll(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculation of NLL assuming noise with zero mean and unit variance.
        
        TODO: confirm we want this.
        """
        # return 0
        return -torch.distributions.Normal(outputs, 1.0).log_prob(targets).sum()

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        # Put model into evaluation mode
        self.eval()

        # Initialise tensor to hold predictions
        output = torch.zeros(size=[len(X), self.output_dim, self.inference_samples]).to(DEVICE)

        # Repeat forward (sampling) <inference_samples> times
        for i in torch.arange(self.inference_samples):
            output[:,:,i] = self.forward(X)
        
        # Determine the average and the variance of the samples
        mean, var = output.mean(dim=-1), output.var(dim=-1)

        return mean, var
