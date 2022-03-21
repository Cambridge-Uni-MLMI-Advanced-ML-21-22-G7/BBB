import os
import logging
from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import Variable
from scipy import interpolate
import pandas as pd

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters, PriorParameters
from bbb.config.constants import KL_REWEIGHTING_TYPES, PRIOR_TYPES, VP_VARIANCE_TYPES, PLOTS_DIR, INFO_DIR
from bbb.models.dnn import RegressionDNN

from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters, PriorParameters
# from bbb.config.constants import (
#     KL_REWEIGHTING_TYPES, PRIOR_TYPES, VP_VARIANCE_TYPES, PLOTS_DIR
# )
from bbb.models.bnn import RegressionBNN
from bbb.data import load_bandit,load_bandit_buffer,load_bandit_train

logger = logging.getLogger(__name__)

fixed_permutation = False
lr = 1e-4
step_size = 5000
gamma = 0.5
idx_list = []
sigma1 = -0
sigma2 = -6

Var = lambda x, dtype=torch.FloatTensor: Variable(
    torch.from_numpy(x).type(dtype)).to(DEVICE)

# coding a class for this
class MushroomBandit(ABC):
    # initializing
    def __init__(self, n_weight_sampling=1):
        self.epsilon = 0
        self.net = None
        self.loss = None
        self.optimizer = None
        self.bufferX = None
        self.bufferY = None
        self.cum_regrets = [0]
        self.n_weight_sampling = n_weight_sampling
        self.pointer = 0

    @property
    def action_eaten(self):
        return torch.Tensor([1,0]).to(DEVICE)

    @property
    def action_noeat(self):
        return torch.Tensor([0,1]).to(DEVICE)

    def get_reward(self,eaten,poison):
        if eaten:
            if poison:
                # Agent eats poison mushroom; 50 % chance of "death"
                # poison = 1, poisonous
                return 5 if np.random.rand() > 0.5 else -35
            else:
                # Agent eats edible mushroom
                return 5
        else:
            # Agent does not eat mushroom
            return 0
    
    def calculate_regret(self, reward, poison):
        """Oracle will always receive a reward of 5 got an edible mushroom
        or zero for a poisonous mushroom.
        """
        if poison:
            return 0 - reward
        else:
            return 5 - reward

    def init_buffer(self, x: torch.Tensor, y: torch.Tensor,z:torch.Tensor):
        self.bufferX = x.clone()
        self.bufferY = y.clone()
        self.bufferZ = z.clone()

    # function to get which mushrooms will be eaten
    def eat_mushrooms(self, X: torch.Tensor, y: torch.Tensor):
        context, poison = X, y

        try_eat = torch.cat((context,self.action_eaten),-1)
        try_reject = torch.cat((context,self.action_noeat),-1)

        with torch.no_grad():
            self.net.eval()
            r_eat = sum([self.net(try_eat) for _ in range(self.n_weight_sampling)]).item()
            r_reject = sum([self.net(try_reject) for _ in range(self.n_weight_sampling)]).item()
        eaten = r_eat > r_reject
        
        if np.random.rand()<self.epsilon:
            eaten = (np.random.rand()<.5)
        agent_reward = self.get_reward(eaten, poison)

        # Get rewards and update buffer
        action = self.action_eaten if eaten else self.action_noeat

        self.bufferX[self.pointer, :-2] = context
        self.bufferX[self.pointer, -2:] = action
        self.bufferY[self.pointer] = agent_reward
        self.bufferZ[self.pointer] = poison.clone()

        if self.pointer >= 4095:
            self.pointer = 0
        else:
            self.pointer += 1

        # Calculate regret
        regret = self.calculate_regret(agent_reward,poison)
        self.cum_regrets.append(self.cum_regrets[-1]+regret)

    # Update buffer
    def update(self, X: torch.Tensor, y: torch.Tensor):
        self.eat_mushrooms(X, y)

        # idx pool
        assert len(self.bufferX) == 4096
        idx_pool = np.random.permutation(len(self.bufferX))
        context_pool = self.bufferX[idx_pool, :]
        reward_pool = self.bufferY[idx_pool]

        avg_loss = 0
        for i in range(0, 4096, 64):
            loss = self.loss_step(context_pool[i:i+64,:], reward_pool[i:i+64], i//64)
            avg_loss = (1/(i+1))*loss + (i/(i+1))*avg_loss
        return avg_loss
    
    @abstractmethod
    def loss_step(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError()

DNN_RL_PARAMETERS = Parameters(
    name = "DNN_rl",
    input_dim = 97,
    output_dim = 1,
    hidden_layers = 3,
    hidden_units = 100,
    batch_size = 64,
    lr = lr,
    gamma = gamma,
    epochs = 50000,
    step_size=step_size,
    early_stopping=False,
    early_stopping_thresh=1e-4
)

# Class for Greedy agents
class Greedy(MushroomBandit):
    def __init__(self, epsilon=0):
        super().__init__()
        self.n_weight_sampling = 1
        self.epsilon = epsilon
        self.net = RegressionDNN(params=DNN_RL_PARAMETERS).to(DEVICE)
        self.criterion = torch.nn.MSELoss()

        with open(os.path.join(self.net.model_save_dir, 'epsilon.txt'), 'w') as f:
            f.write(str(self.epsilon))
        
    def loss_step(self, x, y, batch_id):
        self.net.train()
        self.net.zero_grad()
        preds = self.net(x)
        loss = self.criterion(preds, y)
        loss.backward()
        self.net.optimizer.step()
        return loss.item()


BNN_RL_PARAMETERS = Parameters(
    name = "BBB_rl",
    input_dim = 97,
    output_dim = 1,
    weight_mu_range = [-0.2, 0.2],
    weight_rho_range = [-5, -4],
    regression_likelihood_noise=1.0,
    prior_params = PriorParameters(
        w_sigma=np.exp(sigma1),
        b_sigma=np.exp(sigma1),
        w_sigma_2=np.exp(sigma2),
        b_sigma_2=np.exp(sigma2),
        # w_sigma=1.,
        # b_sigma=1.,
        # w_sigma_2=0.2,
        # b_sigma_2=0.2,
        w_mixture_weight=0.5,
        b_mixture_weight=0.5,
    ),
    hidden_units = 100,
    hidden_layers = 3,
    batch_size = 64,
    lr = lr,
    gamma = gamma,
    epochs=1000,
    step_size=step_size,
    elbo_samples = 2,
    inference_samples = 10,
    prior_type=PRIOR_TYPES.mixture,
    kl_reweighting_type=KL_REWEIGHTING_TYPES.paper,
    vp_variance_type=VP_VARIANCE_TYPES.paper
)

# Class for BBB agents
class BBB_bandit(MushroomBandit):
    def __init__(self):
        super().__init__()
        self.n_weight_sampling = 2
        self.elbo_samples = BNN_RL_PARAMETERS.elbo_samples
        self.net = RegressionBNN(params=BNN_RL_PARAMETERS).to(DEVICE)
        with open(os.path.join(self.net.model_save_dir, 'epsilon.txt'), 'w') as f:
            f.write(str(self.epsilon))
        
    def loss_step(self, x, y, batch_id):
        self.net.train()
        self.net.zero_grad()
        beta = 2 ** (64 - (batch_id + 1)) / (2 ** 64 - 1)
        loss = self.net.sample_ELBO(x, y, beta, self.elbo_samples)
        net_loss = loss[0]
        net_loss.backward()
        self.net.optimizer.step()
        return net_loss.item()


def run_rl_training():
    # Load the data
    
    # Create the directory for storing plots, if it does not already exist
    plot_dir = os.path.join(PLOTS_DIR, 'rl')
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    
    # Create the training information of process, if it does not already exist
    info_dir = os.path.join(INFO_DIR, 'rl')
    if not os.path.isdir(info_dir):
        os.makedirs(info_dir)

    # Define settings
    mnets = {
        'Greedy':Greedy(epsilon=0.0),
        'Greedy 1%':Greedy(epsilon=0.01),
        'Greedy 5%':Greedy(epsilon=0.05),
        'BBB':BBB_bandit()
    }

    NB_STEPS = 50000

    # Initialise buffers
    X,y,z = load_bandit_buffer()
    for name, net in mnets.items():
        net.init_buffer(X, y,z)

    if fixed_permutation:
        X, y = load_bandit_train()
    else:
        X, y = load_bandit()
        idx_permutation = np.random.choice(range(X.shape[0]), size=NB_STEPS, replace=True)
        X = X[idx_permutation, :]
        y = y[idx_permutation]

        assert X.shape[0] == NB_STEPS
        assert y.shape[0] == NB_STEPS

    # Train the RL models
    with tqdm(range(NB_STEPS), unit="batch") as t_epoch:
        for step in t_epoch:
            # mushroom_idx = np.random.randint(X.shape[0])
            # idx_list.append(mushroom_idx)
            for name, net in mnets.items():

                if not step%4096:
                    result = torch.cat((net.bufferX, net.bufferY,net.bufferZ),1)
                    filename = name + '_mix_sigma_{4}_{5}_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,step,sigma1,sigma2) + '.pt'
                    print(filename)
                    torch.save(result, os.path.join(info_dir, filename))

                avg_loss = net.update(X[step], y[step])
                net.net.scheduler.step()

                # Update the loss in tqdm every 10 epochs
                if not step%10:
                    t_epoch.set_postfix_str(f'Loss: {avg_loss:.5f}')
                if not step%5000:
                    ticks = [0, 1000, 10000,100000]
                    fig, ax = plt.subplots() 
                    for name, net in mnets.items():
                        new_y = interpolate.interp1d(ticks, range(4),fill_value="extrapolate")(net.cum_regrets)
                        ax.plot(new_y, label=name)
                    ax.set_xlabel('Steps') 
                    ax.set_ylabel('Regret') 
                    ax.legend()
                    plt.yticks(range(4), ticks)
                    filename = name +'_mix_new' + '_sigma_{4}_{5}_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,step,sigma1,sigma2) + '.jpg'
                    print(filename)
                    plt.savefig(os.path.join(plot_dir, filename))

                    # Save the latest model
                    torch.save(net.net.state_dict(), os.path.join(net.net.model_save_dir, 'model.pt'))
                    

    # Save the cumulative regret for each network
    for name, net in mnets.items():
        np.save(os.path.join(net.net.model_save_dir, 'cum_regrets.npy'), np.array(net.cum_regrets))
        result = torch.cat((net.bufferX, net.bufferY,net.bufferZ),1)
        filename = name +'GPU'+ '_mix_new_sigma_{4}_{5}_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,'final',sigma1,sigma2) + '.pt'
        torch.save(result, os.path.join(info_dir, filename))

    # Plotting
    ticks = [0, 1000, 10000,100000]
    fig, ax = plt.subplots() 
    for name, net in mnets.items():
        new_y = interpolate.interp1d(ticks, range(4))(net.cum_regrets)
        ax.plot(new_y, label=name)
        
    ax.set_xlabel('Steps') 
    ax.set_ylabel('Regret') 
    ax.legend()
    plt.yticks(range(4), ticks)
    
    # Save the plot
    filename = name +'GPU'+  '_mix_new_sigma_{4}_{5}_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,'final',sigma1,sigma2) + '.jpg'
    plt.savefig(os.path.join(plot_dir, filename))

    df = pd.DataFrame(columns=['idx'],data=idx_list)
    filename = name +'GPU'+  '_mix_sigma_{4}_{5}_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,'final',sigma1,sigma2)
    df.to_csv(filename+".csv", encoding='utf-8', index=False)
    # Show the plot
    # plt.show()


if __name__ == '__main__':
    run_rl_training()

