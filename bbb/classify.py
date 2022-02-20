import torch
from models import BNN
from data import load_mnist
from tqdm import tqdm 

import logging

logger = logging.getLogger('__name__')
logging.basicConfig(level=logging.INFO)

params = {
    'name': "test",
    'input_dim': 28*28,
    'output_dim': 10,
    'hidden_units': 1200,
    'weight_mu': [-0.2, 0.2],           # range for mu 
    'weight_rho': [-5, -4],             # range for rho
    'prior_params': {'w_sigma': 1, 'b_sigma': 2},
    'batch_size': 128,
    'lr': 1e-4,
    'epochs': 300,
    'elbo_samples': 2,                     # to draw for ELBO (training)
    'inference_samples': 10,               # to draw for inference
    'save_dir': './saved_models'
}

net = BNN(params=params)

X_train = load_mnist(train=True, batch_size=params['batch_size'], shuffle=True)
X_val = load_mnist(train=False, batch_size=params['batch_size'], shuffle=True)

epochs = params['epochs']
for epoch in tqdm(range(epochs)):
    net.train(X_train)
    
    # If you want to check the parameter values, uncomment the below
    # print(net.optimizer.param_groups)
    
    net.scheduler.step()
    net.eval(X_val)

    logger.info(f'[Epoch {epoch}/{epochs}] - acc: {net.acc}')
    if net.best_acc  and net.acc > net.best_acc:
        net.best_acc = net.acc
        torch.save(net.model.state_dict(), net.save_model_path)
    