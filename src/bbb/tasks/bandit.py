import logging
from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters, PriorParameters
from bbb.config.constants import KL_REWEIGHTING_TYPES, PRIOR_TYPES, VP_VARIANCE_TYPES
from bbb.models.dnn import DNN
from bbb.models.bnn import BanditBNN
from bbb.data import load_bandit


logger = logging.getLogger(__name__)

# coding a class for this
class MushroomBandit(ABC):
    # initializing
    def __init__(self, X, y,n_weight_sampling=2):
        self.epsilon = 0
        self.net = None
        self.loss = None
        self.optimizer = None
        self.bufferX = None
        self.bufferY = None
        self.cum_regrets = [0]
        self.n_weight_sampling = n_weight_sampling

    def get_reward(self,eaten,poison):
        if eaten:
            if poison:
                # poison = 1, poisonous
                return 5 if np.random.rand() > 0.5 else -35
            else:
                return 5
        else:
            return 0
    
    def calculate_regret(self, reward, poison):
        if poison:
            return 0 - reward
        else:
            return 5 - reward

    def init_buffer(self, X: torch.Tensor, y: torch.Tensor):
        self.bufferX = torch.zeros_like(X).to(DEVICE)
        self.bufferY = torch.zeros_like(y).to(DEVICE)

        for i in np.random.choice(range(X.shape[0]), 4096):
            eaten = 1 if np.random.rand() > 0.5 else 0
            self.bufferX[i] = X[i]
            self.bufferY[i] = self.get_reward(eaten, y[i])
        
    # function to get which mushrooms will be eaten
    def eat_mushrooms(self, X: torch.Tensor, y: torch.Tensor, mushroom_idx: int):
        context, poison = X[mushroom_idx], y[mushroom_idx]

        if np.random.rand() < self.epsilon:
            eaten = int(np.random.rand() < 0.5)
        else:
            with torch.no_grad():
                predict_reward = sum([self.net(context) for _ in range(self.n_weight_sampling)]).item()
                eaten = 0 if predict_reward > 0 else 1
        
        agent_reward = self.get_reward(eaten, poison)

        # Get rewards and add these to the buffer
        self.bufferX = torch.vstack((self.bufferX, context))
        self.bufferY = torch.vstack((self.bufferY, torch.Tensor((agent_reward,))))

        # Calculate regret
        regret = self.calculate_regret(agent_reward,poison)
        self.cum_regrets.append(self.cum_regrets[-1]+regret)

    # Update buffer
    def update(self, X: torch.Tensor, y: torch.Tensor, mushroom_idx: int):
        self.eat_mushrooms(X, y, mushroom_idx)

        # idx pool
        l = len(self.bufferX)
        idx_pool = range(l) if l >= 4096 else ((int(4096//l) + 1)*list(range(l)))
        idx_pool = np.random.permutation(idx_pool[-4096:])
        context_pool = self.bufferX[idx_pool, :]
        reward_pool = self.bufferY[idx_pool]

        avg_loss = 0
        for i in range(0, 4096, 64):
            loss = self.loss_step(context_pool[i:i+64], reward_pool[i:i+64], i//64)
            avg_loss = (1/(i+1))*loss + (i/(i+1))*avg_loss
        return avg_loss
    
    @abstractmethod
    def loss_step(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

DNN_REGRESSION_PARAMETERS = Parameters(
    name = "DNN_regression",
    input_dim = 95,
    output_dim = 1,
    hidden_layers = 2,
    hidden_units = 400,
    batch_size = 100,
    lr = 1e-3,
    epochs = 100,
    early_stopping=False,
    early_stopping_thresh=1e-4
)

# Class for Greedy agents
class Greedy(MushroomBandit):
    def __init__(self, epsilon=0, lr=2e-5, **kwargs):
        super().__init__(**kwargs)
        self.n_weight_sampling = 1
        self.epsilon = epsilon
        self.net = DNN(params=DNN_REGRESSION_PARAMETERS).to(DEVICE)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        
    def loss_step(self, x, y, batch_id):
        self.net.zero_grad()
        preds = self.net.forward(x)
        loss = self.criterion(preds, y)
        loss.backward()
        self.optimizer.step()
        return loss

BNN_REGRESSION_PARAMETERS = Parameters(
    name = "BBB_regression",
    input_dim = 95,
    output_dim = 1,
    weight_mu_range = [-1, 1],
    weight_rho_range = [-5, -4],
    prior_params = PriorParameters(
        w_sigma=1.,
        b_sigma=1.,
        w_sigma_2=0.2,
        b_sigma_2=0.2,
        w_mixture_weight=0.5,
        b_mixture_weight=0.5,
    ),
    hidden_units = 400,
    hidden_layers = 2,
    batch_size = 1,
    lr = 1e-3,
    epochs = 100,
    elbo_samples = 2,
    inference_samples = 10,
    prior_type=PRIOR_TYPES.mixture,
    kl_reweighting_type=KL_REWEIGHTING_TYPES.paper,
    vp_variance_type=VP_VARIANCE_TYPES.paper
)

# Class for BBB agents
class BBB_bandit(MushroomBandit):
    def __init__(self, lr=2e-5, **kwargs):
        super().__init__(**kwargs)
        self.n_weight_sampling = 1
        self.net = BanditBNN(params=BNN_REGRESSION_PARAMETERS).to(DEVICE)
        self.n_samples = BNN_REGRESSION_PARAMETERS.name
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        
    def loss_step(self, x, y, batch_id):
        beta = 2 ** (64 - (batch_id + 1)) / (2 ** 64 - 1) 
        self.net.model.train()
        self.net.zero_grad()
        loss = self.net.sample_ELBO(x, y, beta, 2)
        net_loss = loss[0]
        net_loss.backward()
        self.optimizer.step()
        return loss


def run_rl_training():
    # Load the data
    X, y = load_bandit()

    # Define settings
    lr=1e-5
    mnets = {
        # 'Greedy':Greedy(X=X, y=y, lr=lr, epsilon=0),
        # 'Greedy 1%':Greedy(X=X, y=y, lr=lr, epsilon=0.01),
        'Greedy 5%':Greedy(X=X, y=y, lr=lr, epsilon=0.05),
        # 'BBB':BBB_bandit(X=X, y=y, lr=lr)
    }

    NB_STEPS = 5000

    # setting seeds
    # random.seed(123)
    # np.random.seed(123)

    # Initialise buffers
    for name, net in mnets.items():
        net.init_buffer(X, y)

    # Train the RL models
    with tqdm(range(NB_STEPS), unit="batch") as t_epoch:
        for step in t_epoch:
            mushroom_idx = np.random.randint(X.shape[0])
            for name, net in mnets.items():
                avg_loss = net.update(X, y, mushroom_idx)

                # Update the loss in tqdm every 10 epochs
                if not step%10:
                    t_epoch.set_postfix_str(f'Loss: {avg_loss:.5f}')

    # Plotting
    fig, ax = plt.subplots() 
    for name, net in mnets.items():
        ax.plot(net.cum_regrets, label=name)
    ax.set_xlabel('Steps') 
    ax.set_ylabel('Regret') 
    ax.legend()
    plt.show()


if __name__ == '__main__':
    run_rl_training()

