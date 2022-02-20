import os
from random import sample
from tqdm import tqdm 
import torch
from torch import nn
from layers import BFC
import logging

logger = logging.getLogger(__name__)

class BNN(nn.Module):
    """ Bayesian (Weights) Neural Network """
    def __init__(self, params) -> None:
        super().__init__()

        # Parameters
        self.input_dim = params['input_dim'] # params.get('input_dim', "default value")
        self.hidden_units = params['hidden_units']
        self.output_dim = params['output_dim']
        self.weight_mu = params['weight_mu']
        self.weight_rho = params['weight_rho']
        self.prior_params = params['prior_params']
        self.elbo_samples = params['elbo_samples'] # num samples to draw for ELBO
        self.inference_samples = params['inference_samples'] # num samples to draw for ELBO
        self.batch_size = params['batch_size']
        self.lr = params['lr']

        # Logs
        self.name = params['name']
        self.best_acc = None
        self.model_path = f'{params["save_dir"]}/{params["name"]}_model.pt'
        if not os.path.exists(params['save_dir']):
            os.makedirs(params['save_dir'])
        

        # Model
        self.model = nn.Sequential(
                    BFC(self.input_dim, self.hidden_units, self.weight_mu, self.weight_rho, self.prior_params), 
                    nn.ReLU(),
                    BFC(self.hidden_units, self.hidden_units, self.weight_mu, self.weight_rho, self.prior_params),
                    nn.ReLU(),
                    BFC(self.hidden_units, self.output_dim, self.weight_mu, self.weight_rho, self.prior_params))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100, gamma=0.5)


    def forward(self, x):
        x = x.view(-1, self.input_dim) # flatten images, necessary for classification
        return self.model.forward(x) 

    def inference(self, x):
        """ Here we do not draw weights but take the mean """
        x = x.view(-1, self.input_dim)
        for layer in self.model:
            if layer == BFC:
                x = layer.forward(x, sample=False)
            else:
                x = layer.forward(x)
        return x

    def log_prior(self):
        log_prior = 0
        for layer in self.model:
            if type(layer) == BFC:
                log_prior += layer.log_prior

        return log_prior

    def log_posterior(self):
        log_posterior = 0
        for layer in self.model:
            if type(layer) == BFC:
                log_posterior += layer.log_posterior

        return log_posterior

    def get_nll(self, outputs, targets):
        nll = nn.CrossEntropyLoss(reduction='sum')(outputs, targets) # classification
        return nll

    def sample_ELBO(self, x, y, beta, num_samples):
        """ run X through the (sampled) model <samples> times"""
        log_priors = torch.zeros(num_samples)
        log_variational_posteriors = torch.zeros(num_samples)
        nll = torch.zeros(1)

        for i in range(num_samples):
            preds = self.inference(x)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_posterior()
            nll += self.get_nll(preds, y)

        # Compute loss
        log_prior = beta*log_priors.mean()          # why is there a beta?
        log_variational_posterior = beta*log_variational_posteriors.mean() # why is there a beta?
        nll /= num_samples
        loss = log_variational_posterior - log_prior + nll
        return loss, log_priors.mean(), log_variational_posteriors.mean(), nll


    def train(self, train_data):
        self.model.train()

        for idx, (X, Y) in enumerate(tqdm(train_data)):
            beta = 2 ** (len(train_data) - (idx + 1)) / (2 ** len(train_data) - 1) 
            self.model.zero_grad()
            self.loss_info = self.sample_ELBO(X, Y, beta, self.elbo_samples)            
            
            model_loss = self.loss_info[0] # loss = kl + nll
            model_loss.backward(retain_graph=True)          # called on the last layer, need retain_graph for some reason
            
            # To inspect optimizer loss, which currently always returns None (problematic)
            # logger.info("\nOptimizer loss: {}\n".format(self.optimizer.step()))


    def predict(self, X):
        probs = torch.zeros(size=[len(X), self.output_dim])

        # Repeat forward (sampling) <inference_samples> times to create probability distrib
        for _ in torch.arange(self.inference_samples):
            output = self.forward(X)
            out = torch.nn.Softmax(dim=1)(output)
            probs += out / self.inference_samples
        
        # Select most likely class
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs

    def eval(self, test_data):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(test_data):
                X, y = data
                preds, _ = self.predict(X)
                total += self.batch_size
                correct += (preds == y).sum().item()
        
        self.acc = correct / total
        if self.best_acc == None: self.best_acc = self.acc
        # logger.info(f'{self.name} validation accuracy: {self.acc}')
