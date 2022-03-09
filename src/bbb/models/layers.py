import logging

import torch
from torch import nn, distributions
from torch.nn import Parameter

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.constants import PRIOR_TYPES, VP_VARIANCE_TYPES
from bbb.config.parameters import PriorParameters


logger = logging.getLogger(__name__)

class GaussianVarPost(nn.Module):
    def __init__(self, mu: float, rho: float, vp_var_type: int, dim_in: int = None, dim_out: int = None) -> None:
        super().__init__()

        if dim_in == None: # bias tensor
            mu_tensor = torch.Tensor(dim_out).uniform_(*mu)
            rho_tensor = torch.Tensor(dim_out).uniform_(*rho)
        else:
            # Return torch.mm(input, w) + b  in fwd pass if using commented out lines
            # mu_tensor = torch.Tensor(dim_in, dim_out).uniform_(*mu)
            # rho_tensor = torch.Tensor(dim_in, dim_out).uniform_(*rho)

            mu_tensor = torch.Tensor(dim_out, dim_in).uniform_(*mu)
            rho_tensor = torch.Tensor(dim_out, dim_in).uniform_(*rho)
        self.mu = Parameter(mu_tensor)
        self.rho = Parameter(rho_tensor)
        self.vp_var_type = vp_var_type

    @property
    def sigma(self):
        if self.vp_var_type == VP_VARIANCE_TYPES.paper:
            return torch.log1p(1+torch.exp(self.rho))  # section 3.2
        elif self.vp_var_type == VP_VARIANCE_TYPES.simple:
            return torch.log1p(torch.exp(self.rho))
        else:
            raise RuntimeError(f'Unrecognised variational posterior variance type: {self.vp_var_type}')

    def sample(self):
        epsilon = distributions.Normal(0,1).sample(self.rho.size()).to(DEVICE)
        sample = self.mu + self.sigma * epsilon
        return sample

    def log_prob(self, value):
        print(self.mu)
        log_prob = distributions.Normal(loc=self.mu, scale=self.sigma).log_prob(value)
        return log_prob


class BFC(nn.Module):
    """Bayesian (Weights) Fully Connected Layer"""
    
    def __init__(
        self,
        dim_in: int, 
        dim_out: int,
        weight_mu: float,
        weight_rho: float,
        prior_params: PriorParameters,
        prior_type: int,
        vp_var_type: int,
    ):
        super().__init__()
        
        # Create IN X OUT weight tensor that we can sample from
        # This is the variational posterior over the weights
        self.w_var_post = GaussianVarPost(weight_mu, weight_rho, dim_in=dim_in, dim_out=dim_out, vp_var_type=vp_var_type)
        self.b_var_post = GaussianVarPost(weight_mu, weight_rho, dim_out=dim_out, vp_var_type=vp_var_type)

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
            logger.info(f'Weights Prior: Gaussian mixture with means {(0,0)}, variances {(prior_params.w_sigma, prior_params.w_sigma_2)} and weight {prior_params.w_mixture_weight}')
            logger.info(f'Biases Prior: Gaussian mixture with means {(0,0)}, variances {(prior_params.b_sigma, prior_params.b_sigma_2)} and weight {prior_params.b_mixture_weight}')

            assert all((prior_params.w_sigma_2, prior_params.b_sigma_2, prior_params.w_mixture_weight, prior_params.b_mixture_weight))
            
            w_mix = distributions.Categorical(torch.tensor((prior_params.w_mixture_weight, 1-prior_params.w_mixture_weight)))
            b_mix = distributions.Categorical(torch.tensor((prior_params.b_mixture_weight, 1-prior_params.b_mixture_weight)))

            w_norm_comps = distributions.Normal(torch.zeros(2), torch.tensor((prior_params.w_sigma, prior_params.w_sigma_2), dtype=torch.float32))
            b_norm_comps = distributions.Normal(torch.zeros(2), torch.tensor((prior_params.b_sigma, prior_params.b_sigma_2), dtype=torch.float32))

            self.w_prior = distributions.MixtureSameFamily(w_mix, w_norm_comps)
            self.b_prior = distributions.MixtureSameFamily(b_mix, b_norm_comps)
        else:
            raise RuntimeError(f'Unexpected prior type: {prior_type}')

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
