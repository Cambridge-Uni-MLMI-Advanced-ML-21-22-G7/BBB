import logging
from abc import ABC, abstractmethod
from time import sleep
from typing import Tuple, List, Union

import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.constants import KL_REWEIGHTING_TYPES, PRIOR_TYPES, VP_VARIANCE_TYPES
from bbb.config.parameters import Parameters
from bbb.models.base import BaseModel
from bbb.models.layers import BFC, BFC_LRT, BaseBFC
from bbb.models.evaluation import RegressionEval, ClassificationEval


logger = logging.getLogger(__name__)


class BaseBNN(BaseModel, ABC):
    """Bayesian (Weights) Neural Network
    
    This class inherits from BaseModel, and is inherited by specific
    Regression and Classification classes. See below.
    """
    def __init__(self, params: Parameters, eval_mode: bool = False) -> None:
        super().__init__(params=params, eval_mode=eval_mode)

        self.batch_size = params.batch_size
        # Architecture
        self.input_dim = params.input_dim
        self.hidden_units = params.hidden_units
        self.hidden_layers = params.hidden_layers
        self.output_dim = params.output_dim
        # Paper Choices
        self.prior_type = params.prior_type
        self.kl_reweighting_type = params.kl_reweighting_type
        self.vp_variance_type = params.vp_variance_type
        self.local_reparam_trick = params.local_reparam_trick
        # VI
        self.weight_mu_range = params.weight_mu_range
        self.weight_rho_range = params.weight_rho_range
        self.prior_params = params.prior_params
        self.elbo_samples = params.elbo_samples
        self.inference_samples = params.inference_samples
        # Optimiser
        self.opt_choice = params.opt_choice
        self.lr = params.lr
        # LR Scheduler
        self.step_size = params.step_size
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

        # Define layer type
        if self.local_reparam_trick:
            BFC_CLASS = BFC_LRT
        else:
            BFC_CLASS = BFC

        # Model
        model_layers = []

        # Input layer
        model_layers.append(BFC_CLASS(
            dim_in=self.input_dim,
            dim_out=self.hidden_units,
            **bfc_arguments)
        )
        model_layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(self.hidden_layers-2):
            model_layers.append(BFC_CLASS(
                dim_in=self.hidden_units,
                dim_out=self.hidden_units,
                **bfc_arguments
            ))
            model_layers.append(nn.ReLU())
        
        # Final output layer
        model_layers.append(BFC_CLASS(
            dim_in=self.hidden_units,
            dim_out=self.output_dim,
            **bfc_arguments)
        )
        self.model = nn.Sequential(*model_layers)

        # Optimizer
        self.optimizer = getattr(optim, self.opt_choice)(
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
            if isinstance(layer, BaseBFC):
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
            if isinstance(layer, BaseBFC):
                log_prior += layer.log_prior

        return log_prior

    def log_var_posterior(self) -> float:
        """Calculate the log variational posterior; log q(w|theta).

        :return: the log prior
        :rtype: float
        """
        log_posterior = 0
        for layer in self.model:
            if isinstance(layer, BaseBFC):
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
            # DO NOT CHANGE CHECK FROM BFC_LRT - KL divergence only applies
            # when using the local reparameterisation trick
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
            elif self.kl_reweighting_type == KL_REWEIGHTING_TYPES.paper_inv:
                pi = 2 ** idx / (2 ** num_batches - 1)
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
        
        # Step the scheduler forward
        self.scheduler.step()

        # Record the ELBO
        self.loss_hist.append(batch_elbo.item())

        # Return the ELBO figure of the final batch as a representative example
        return batch_elbo.item()

    def get_weights(self) -> Union[List[Tensor], List[Tensor]]:
        """Obtain the mean and standard deviation tensors of the weights in
        each layer.

        :return: list of mean and standard deviation tensors
        :rtype: Union[List[Tensor], List[Tensor]]
        """
        weight_means = []
        weight_stds = []
        
        for layer in [l for l in self.model if isinstance(l, BaseBFC)]:
            weight_means.append(layer.w_var_post.to(DEVICE).mu)
            weight_stds.append(layer.w_var_post.to(DEVICE).sigma)
        
        return weight_means, weight_stds

    def prune_weights(self, weight_fill_masks: List[Tensor]):
        """Perform a masked fill of the mean and rho tensors. Effectively we are
        pruning those weights that are True in the mask.

        :param weight_fill_masks: list of masks, one for each layer
        :type weight_fill_masks: List[Tensor]
        """

        # The value to set rho to depends on the variance type employed
        # We want the resulting varaince to be 0
        if self.vp_variance_type == VP_VARIANCE_TYPES.paper:
            rho_replacement = -np.inf
        elif self.vp_variance_type == VP_VARIANCE_TYPES.simple:
            rho_replacement = 0

        # Perform a masked fill, using the list of weight_fill_masks
        for i, layer in enumerate([l for l in self.model if isinstance(l, BaseBFC)]):
            layer.w_var_post.mu = nn.Parameter(
                layer.w_var_post.mu.masked_fill(weight_fill_masks[i], 0)
            )
            layer.w_var_post.rho = nn.Parameter(
                layer.w_var_post.rho.masked_fill(weight_fill_masks[i], rho_replacement)
            )

    def get_pruned_weight_samples(self, weight_fill_masks):
        """ Returns the samples of weights that are active post-pruning """
        bfc_layers = [layer for layer in self.model if isinstance(layer, BaseBFC)]
        non_pruned_samples = []
    
        weight_samples = self.weight_samples()
        for layer_samples, layer, layer_mask in zip(weight_samples, bfc_layers, weight_fill_masks):
        
            ls = torch.reshape(layer_samples, shape=(self.inference_samples, layer.dim_in, layer.dim_out))
            non_pruned_samples_layer = []
        
            for i, sample in enumerate(ls):
                non_pruned_weights = sample[~layer_mask.T]
                num_non_pruned_weights, num_weights = non_pruned_weights.shape[0], sample.shape[0]*sample.shape[1]
                non_pruned_samples_layer.append(non_pruned_weights.flatten())

            non_pruned_samples.append(torch.flatten(torch.stack(non_pruned_samples_layer)))

        return non_pruned_samples

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
    
        # Input layers
        weight_tensors = [
            torch.zeros(size=(self.input_dim*self.hidden_units, self.inference_samples)).to(DEVICE)]

        # Hidden layers
        weight_tensors.extend([
            torch.zeros(size=(self.hidden_units**2, self.inference_samples)).to(DEVICE)
            for _ in range(self.hidden_layers-2)
        ])

        # Output layer
        weight_tensors.append(torch.zeros(size=(self.hidden_units*self.output_dim, self.inference_samples)).to(DEVICE))

        # Repeat sampling <inference_samples> times
        for i, layer in enumerate([l for l in self.model if isinstance(l, BaseBFC)]):
            for j in range(self.inference_samples):
                weight_tensors[i][:, j] = layer.w_var_post.to(DEVICE).sample().flatten()
            
            # Take the mean across the samples
            # weight_tensors[i] = weight_tensors[i].mean(axis=-1)

            # Take all samples
            weight_tensors[i] = weight_tensors[i].flatten()


        ####################
        # Using mean weights
        ####################
        # weight_tensors = []
        # for layer in [l for l in self.model if isinstance(l, BaseBFC)]:

            ################
            # Sampling from N(mu, sigma)
            ################
            # mu = layer.w_var_post.mu 
            # rho = layer.w_var_post.rho 
            # sigma = torch.log1p(torch.exp(rho))
    
            # draws = torch.normal(mu, sigma)
            # weight_tensors.append(draws.flatten())

            ###############
            # Return mu directly
            ###############
            # weight_tensors.append(layer.w_var_post.mu.flatten())

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
        q = torch.tensor([0.05, 0.25, 0.75, 0.95]).to(DEVICE)
        quartiles = torch.quantile(output, q, dim=-1)

        return mean, var, quartiles

    def lpd(self, X: Tensor, Y: Tensor):
        """Calculate the log predictive density of the model.

        https://vasishth.github.io/bayescogsci/book/expected-log-predictive-density-of-a-model.html
        """
        # Ensure tensor is assigned to correct device
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        # Put model into evaluation mode
        self.eval()

        # Initialise tensor to hold predictions
        lpd = 0

        with torch.no_grad():
            # Repeat forward (sampling) <inference_samples> times
            for _ in torch.arange(self.inference_samples):
                preds = self.forward(X)
                lpd += (1/self.inference_samples) * torch.distributions.Normal(preds, torch.ones(len(preds))*self.regression_likelihood_noise).log_prob(Y).mean()

        return lpd

class ClassificationBNN(ClassificationEval, BaseBNN):
    # NOTE: This class inherits from ClassificationEval and then BaseBNN
    # The order here is important

    def forward(self, X: Tensor) -> Tensor:
        # Flatten the image
        X = X.view(-1, self.input_dim)
        return super().forward(X)

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
            output = F.softmax(self.forward(X), dim=1)

            # Incremental update of average
            probs += output / self.inference_samples
        
        # Select most likely class
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
