import logging
from typing import List

import torch
from torch import nn, distributions, Tensor
from torch.nn import Parameter

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.constants import PRIOR_TYPES, VP_VARIANCE_TYPES
from bbb.config.parameters import PriorParameters


logger = logging.getLogger(__name__)

class GaussianVarPost(nn.Module):
    def __init__(self, mu: float, rho: float, vp_var_type: int, dim_in: int = None, dim_out: int = None) -> None:
        super().__init__()

        # Initialisation of the weight and bias arrays/tensors
        # Remember that it is these that we are going to learn.
        # TODO: ask @Max why we initialise these using uniform dist - where did he read about this?
        if dim_in == None: # bias tensor
            mu_tensor = torch.Tensor(dim_out).uniform_(*mu)
            rho_tensor = torch.Tensor(dim_out).uniform_(*rho)
        else:
            # Return torch.mm(input, w) + b  in fwd pass if using commented out lines
            # mu_tensor = torch.Tensor(dim_in, dim_out).uniform_(*mu)
            # rho_tensor = torch.Tensor(dim_in, dim_out).uniform_(*rho)

            mu_tensor = torch.Tensor(dim_out, dim_in).uniform_(*mu)
            rho_tensor = torch.Tensor(dim_out, dim_in).uniform_(*rho)

        # nn.init.constant_(mu_tensor, mu)
        # nn.init.constant_(rho_tensor, rho)
        self.mu = Parameter(mu_tensor)
        self.rho = Parameter(rho_tensor)
        self.vp_var_type = vp_var_type
        self.std_normal = distributions.Normal(0,1)

    @property
    def sigma(self):
        if self.vp_var_type == VP_VARIANCE_TYPES.paper:
            return torch.log1p(torch.exp(self.rho))  # section 3.2
        elif self.vp_var_type == VP_VARIANCE_TYPES.simple:
            return torch.log(torch.exp(self.rho))
        else:
            raise RuntimeError(f'Unrecognised variational posterior variance type: {self.vp_var_type}')

    def sample(self):
        """Equivalent to using a factorised Gaussian posterior over the weights.
        Independent samples drawn from standard normal distribution, and then the reparameterisation
        trick is applied.

        Not that we draw these weight samples once for each minibatch, which leads to correlations
        between the datapoints (given the same weight samples are used when determining the activation
        for each datapoint). The local reparameterisation trick seeks to get around this by sampling
        activations conditional on the datapoints, rather than sampling weights.
        """
        epsilon = self.std_normal.sample(self.rho.size()).to(DEVICE)
        sample = self.mu + self.sigma * epsilon
        return sample

    def log_prob(self, value):
        log_prob = distributions.Normal(loc=self.mu, scale=self.sigma).log_prob(value)
        return log_prob


class BaseBFC(nn.Module):
    """Bayesian (Weights) Fully Connected Layer"""
    
    def __init__(
        self,
        dim_in: int, 
        dim_out: int,
        weight_mu_range: List[float],
        weight_rho_range: List[float],
        prior_params: PriorParameters,
        prior_type: int,
        vp_var_type: int,
    ):
        super().__init__()
        
        # Create IN X OUT weight tensor that we can sample from
        # This is the variational posterior over the weights
        self.w_var_post = GaussianVarPost(weight_mu_range, weight_rho_range, dim_in=dim_in, dim_out=dim_out, vp_var_type=vp_var_type)
        self.b_var_post = GaussianVarPost(weight_mu_range, weight_rho_range, dim_out=dim_out, vp_var_type=vp_var_type)

        # Set Prior distribution over the weights and biases
        assert prior_params.w_sigma and prior_params.b_sigma  # Assert that minimum required prior parameters are present
        if prior_type == PRIOR_TYPES.single:
            # Single Gaussian distribution
            logger.info(f'Weights Prior: Gaussian with mean {0} and variance {prior_params.w_sigma}')
            logger.info(f'Biases Prior: Gaussian with mean {0} and variance {prior_params.b_sigma}')

            self.w_prior = distributions.Normal(0, prior_params.w_sigma)
            self.b_prior = distributions.Normal(0, prior_params.b_sigma)
        elif prior_type == PRIOR_TYPES.mixture:
            # Mixture of Gaussian distributions
            # Implemented using the PyTorch MixtureSameFamily distribution
            # https://pytorch.org/docs/stable/distributions.html#mixturesamefamily
            logger.info(f'Weights Prior: Gaussian mixture with means {(0,0)}, variances {(prior_params.w_sigma, prior_params.w_sigma_2)} and weight {prior_params.w_mixture_weight}')
            logger.info(f'Biases Prior: Gaussian mixture with means {(0,0)}, variances {(prior_params.b_sigma, prior_params.b_sigma_2)} and weight {prior_params.b_mixture_weight}')

            # Ensure that all the necessary parameters have been provided for a GMM
            assert all((prior_params.w_sigma_2, prior_params.b_sigma_2, prior_params.w_mixture_weight, prior_params.b_mixture_weight))

            # Specify the desired weights
            w_mix = distributions.Categorical(torch.tensor((prior_params.w_mixture_weight, 1-prior_params.w_mixture_weight), device=DEVICE))
            b_mix = distributions.Categorical(torch.tensor((prior_params.b_mixture_weight, 1-prior_params.b_mixture_weight), device=DEVICE))

            # Specify the individual components - whilst these appear to be multivariate Gaussians they will be seperated
            w_norm_comps = distributions.Normal(torch.zeros(2, device=DEVICE), torch.tensor((prior_params.w_sigma, prior_params.w_sigma_2), device=DEVICE, dtype=torch.float32))
            b_norm_comps = distributions.Normal(torch.zeros(2, device=DEVICE), torch.tensor((prior_params.b_sigma, prior_params.b_sigma_2), device=DEVICE, dtype=torch.float32))

            # Create the GMMs
            self.w_prior = distributions.MixtureSameFamily(w_mix, w_norm_comps)
            self.b_prior = distributions.MixtureSameFamily(b_mix, b_norm_comps)
        elif prior_type == PRIOR_TYPES.laplacian:
            # Single Laplacian distribution
            logger.info(f'Weights Prior: Laplacian with mean {0} and variance {prior_params.w_sigma}')
            logger.info(f'Biases Prior: Laplacian with mean {0} and variance {prior_params.b_sigma}')

            self.w_prior = distributions.Laplace(0, prior_params.w_sigma)
            self.b_prior = distributions.Laplace(0, prior_params.b_sigma)
        else:
            raise RuntimeError(f'Unexpected prior type: {prior_type}')

class BFC(BaseBFC):
    """Bayesian (Weights) Fully Connected Layer"""
    
    def __init__(
        self,
        dim_in: int, 
        dim_out: int,
        weight_mu_range: List[float],
        weight_rho_range: List[float],
        prior_params: PriorParameters,
        prior_type: int,
        vp_var_type: int,
    ):
        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            weight_mu_range=weight_mu_range,
            weight_rho_range=weight_rho_range,
            prior_params=prior_params,
            prior_type=prior_type,
            vp_var_type=vp_var_type
        )

        self.log_prior = 0
        self.log_var_post = 0

    def forward(self, input, sample=True):
        """For every forward pass in training, we are drawing weights (incl. bias) from the variational posterior and return a determinstic forward mapping"""
        
        if self.training or sample:
            # Draw weights from the variational posterior
            w = self.w_var_post.sample()
            b = self.b_var_post.sample()

            # Compute free energy
            self.log_prior = self.w_prior.log_prob(w).sum() + self.b_prior.log_prob(b).sum() # log P(w) in eq (2)
            self.log_var_post = self.w_var_post.log_prob(w).sum() + self.b_var_post.log_prob(b).sum()  # log q(w|theta) in eq (2)

        else:
            w = self.w_var_post.mu
            b = self.b_var_post.mu

            self.log_prior = 0
            self.log_posterior = 0
            
        # Get weight data
        # print("input: ", input.size())
        # print("w: ", w.size()) 
        # print("b: ", b.size()) 
    
        # return torch.mm(input, w) + b # (IF you specify weights above as Tensor(dim_out, dim_in))
        return nn.functional.linear(input, w, b)

class BFC_LRT(BaseBFC):
    """Bayesian (Weights) Fully Connected Layer using the local reparameterisation trick"""
    
    def __init__(
        self,
        dim_in: int, 
        dim_out: int,
        weight_mu_range: List[float],
        weight_rho_range: List[float],
        prior_params: PriorParameters,
        prior_type: int,
        vp_var_type: int,
    ):
        # If using local reparameterisation trick the prior must be Gaussian
        # This is due to the exact calculation of the KL divergence
        # KL(q(w)||p(w))
        assert prior_type == PRIOR_TYPES.single

        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            weight_mu_range=weight_mu_range,
            weight_rho_range=weight_rho_range,
            prior_params=prior_params,
            prior_type=prior_type,
            vp_var_type=vp_var_type
        )
        
        self.kl_d = 0

    def forward(self, input, sample=True):
        """For every forward pass in training, we are drawing samples from a distribution over the activation function.
        """
        
        if self.training or sample:
            gamma = nn.functional.linear(input, self.w_var_post.mu)
            delta = torch.sqrt(1e-32 + nn.functional.linear(input.pow(2), self.w_var_post.sigma.pow(2)))

            w_zeta = distributions.Normal(0,1).sample(gamma.size()).to(DEVICE)
            b_zeta = distributions.Normal(0,1).sample(self.b_var_post.mu.size()).to(DEVICE)

            w_act_sample = gamma + delta * w_zeta
            b_act_sample = self.b_var_post.mu + self.b_var_post.sigma * b_zeta

            self.kl_d = self._kl_d()

            activation = w_act_sample + b_act_sample
            return activation
        else:
            raise NotImplementedError(
                "Not yet implemented."
            )
            

    def _calc_kl_d(self, mu_q: Tensor, sigma_q: Tensor, mu_p: Tensor, sigma_p: Tensor) -> float:
        """Determine KL(q||p) as follows (I've confirmed this before in the past):
        https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

        log(\sigma_p/\sigma_q) + ((\sigma_q^2+(\mu_q-\mu_p)^2)/(2*\sigma_p^2)) - 1/2

        :param mu_q: _description_
        :type mu_q: Tensor
        :param sigma_q: _description_
        :type sigma_q: Tensor
        :param mu_p: _description_
        :type mu_p: Tensor
        :param sigma_p: _description_
        :type sigma_p: Tensor
        :return: _description_
        :rtype: Tensor
        """
        return 0.5 * (2 * torch.log(sigma_p / sigma_q) - 1 + (sigma_q / sigma_p).pow(2) + ((mu_p - mu_q) / sigma_p).pow(2)).sum()
        # return (torch.log((sigma_p/sigma_q)) + ((sigma_q.pow(2) + (mu_q-mu_p).pow(2))/(2*sigma_p.pow(2))) - 0.5).sum()

    def _kl_d(self) -> float:
        """Determine and add the KL divergence for weights and biases.

        :return: KL divergence
        :rtype: float
        """
        w_kld = self._calc_kl_d(
            mu_q=self.w_var_post.mu,
            sigma_q=self.w_var_post.sigma,
            mu_p=self.w_prior.mean,
            sigma_p=self.w_prior.stddev
        )
        b_kld = self._calc_kl_d(
            mu_q=self.b_var_post.mu,
            sigma_q=self.b_var_post.sigma,
            mu_p=self.b_prior.mean,
            sigma_p=self.b_prior.stddev
        )
        return w_kld + b_kld
