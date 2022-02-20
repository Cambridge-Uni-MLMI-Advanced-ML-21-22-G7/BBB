from torch import distributions
from torch import nn
from torch.nn import Parameter
import torch
import math


class BayesianWeight(nn.Module):
    def __init__(self, mu, rho, dim_in=None, dim_out=None) -> None:
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
        self.sigma = torch.log1p(torch.exp(self.rho)) # section 3.2: \sigma = log(1+exp(\rho))
        self.dist = distributions.Normal(loc=self.mu, scale=self.sigma)

    def sample(self):
        return self.dist.sample()

    def log_prob(self, value):
        return self.dist.log_prob(value)

class BFC(nn.Module):
    """ Bayesian (Weights) Fully Connected Layer """
    
    def __init__(self, dim_in, dim_out, weight_mu, weight_rho, prior_params):
        super().__init__()
        
        # Create IN X OUT weight tensor that we can sample from 
        self.weights = BayesianWeight(weight_mu, weight_rho, dim_in=dim_in, dim_out=dim_out)
        self.bias = BayesianWeight(weight_mu, weight_rho, dim_out=dim_out)

        # Set prior
        assert prior_params.w_sigma and prior_params.b_sigma  # Assert that the required prior parameters are present
        self.w_prior = distributions.Normal(0, prior_params.w_sigma)
        self.b_prior = distributions.Normal(0, prior_params.b_sigma)

        self.log_prior = 0
        self.log_posterior = 0

    def forward(self, input, sample=True):
        """ For every forward pass in training, we are drawing weights (incl. bias) and return a determinstic forward mapping"""
        
        if self.training or sample:
            # draw weights
            w = self.weights.sample()
            b = self.bias.sample()

            # Compute free energy
            self.log_prior = self.w_prior.log_prob(w).sum() + self.b_prior.log_prob(b).sum() # log P(w) in eq (2)
            self.log_posterior = self.weights.log_prob(w).sum() + self.bias.log_prob(b).sum()  # log q(w|theta) in eq (2)

        else:
            w = self.weights.mu
            b = self.bias.mu

            self.log_prior = 0
            self.log_posterior = 0
            
        # Get weight data
        # print("input: ", input.size())
        # print("w: ", w.size()) 
        # print("b: ", b.size()) 
    
        # return torch.mm(input, w) + b # (IF you specify weights above as Tensor(dim_out, dim_in))
        return nn.functional.linear(input, w, b)
