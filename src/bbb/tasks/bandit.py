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
from bbb.models.bnn import BanditBNN
from bbb.data import load_bandit,load_bandit_buffer,load_bandit_train

logger = logging.getLogger(__name__)
lr = 1e-4
step_size = 5001
gamma = 0.5
idx_list = []

Var = lambda x, dtype=torch.FloatTensor: Variable(
    torch.from_numpy(x).type(dtype)).to(DEVICE)

# coding a class for this
class MushroomBandit(ABC):
    # initializing
    def __init__(self, n_weight_sampling=2):
        self.epsilon = 0
        self.net = None
        self.loss = None
        self.optimizer = None
        self.bufferX = None
        self.bufferY = None
        self.cum_regrets = [0]
        self.n_weight_sampling = n_weight_sampling
        self.action_eaten = torch.Tensor([1,0]).to(DEVICE)
        self.action_noeat = torch.Tensor([0,1]).to(DEVICE)
        self.pointer = 0

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

    def init_buffer(self, x:torch.Tensor, y: torch.Tensor):
        self.bufferX = torch.empty(4096, 97).to(DEVICE)
        self.bufferY = torch.empty(4096, 1).to(DEVICE)
        self.bufferZ = torch.empty(4096, 1).to(DEVICE)
        
        for i, idx in enumerate(np.random.choice(range(x.shape[0]), 4096)):
            eaten = 1 if np.random.rand() > 0.5 else 0
            action = self.action_eaten if eaten else self.action_noeat
            self.bufferX[i] = torch.cat((x[idx],action),-1)
            self.bufferY[i] = self.get_reward(eaten, y[idx])
            self.bufferZ[i] = y[idx]
        
    # function to get which mushrooms will be eaten
    def eat_mushrooms(self, X: torch.Tensor, y: torch.Tensor, mushroom_idx: int):
        context, poison = X[mushroom_idx], y[mushroom_idx]

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

        # Get rewards and add these to the buffer
        # self.bufferX = torch.vstack((self.bufferX, torch.cat((context,action),-1)))
        # self.bufferY = torch.vstack((self.bufferY, torch.Tensor((agent_reward,)).to(DEVICE)))
        # self.bufferZ = torch.vstack((self.bufferZ, torch.Tensor((poison,)).to(DEVICE)))

        self.bufferX[self.pointer] = torch.cat((context,action),-1)
        self.bufferY[self.pointer] = torch.Tensor((agent_reward,)).to(DEVICE)
        self.bufferZ[self.pointer] = poison
        if self.pointer >= 4095:
            self.pointer = 0
        else:
            self.pointer += 1

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

DNN_RL_PARAMETERS = Parameters(
    name = "DNN_rl",
    input_dim = 97,
    output_dim = 1,
    hidden_layers = 3,
    hidden_units = 100,
    batch_size = 64,
    lr = lr,
    gamma = gamma,
    epochs = 1000,
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
        self.net.zero_grad()
        preds = self.net.forward(x)
        loss = self.criterion(preds, y)
        loss.backward()
        self.net.optimizer.step()
        return loss


BNN_RL_PARAMETERS = Parameters(
    name = "BBB_rl",
    input_dim = 97,
    output_dim = 1,
    weight_mu_range = [-0.2, 0.2],
    weight_rho_range = [-5, -4],

    prior_params = PriorParameters(
        w_sigma=np.exp(-0),
        b_sigma=np.exp(-0),
        w_sigma_2=np.exp(-6),
        b_sigma_2=np.exp(-6),
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
    epochs=50000,
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
        self.net = BanditBNN(params=BNN_RL_PARAMETERS).to(DEVICE)

        with open(os.path.join(self.net.model_save_dir, 'epsilon.txt'), 'w') as f:
            f.write(str(self.epsilon))
        
    def loss_step(self, x, y, batch_id):
        self.net.eval()
        self.net.model.eval()
        beta = 2 ** (64 - (batch_id + 1)) / (2 ** 64 - 1)
        beta = torch.Tensor((beta,)).to(DEVICE)
        self.net.optimizer.zero_grad()
        num_samples = 2
        loss = self.net.sample_ELBO(x, y, beta,num_samples)
        net_loss = loss[0]
        net_loss.backward()
        self.net.optimizer.step()
        return net_loss.item()


def run_rl_training():
    # Load the data
    X, y = load_bandit()
    
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
        # 'Greedy':Greedy(epsilon=0),
        # 'Greedy 1%':Greedy(epsilon=0.01),
        # 'Greedy 5%':Greedy(epsilon=0.05),
        'BBB':BBB_bandit()
    }

    NB_STEPS = 50000

    # Initialise buffers
    for name, net in mnets.items():
        net.init_buffer(X, y)

    # Train the RL models
    with tqdm(range(NB_STEPS), unit="batch") as t_epoch:
        for step in t_epoch:
            mushroom_idx = np.random.randint(X.shape[0])
            idx_list.append(mushroom_idx)
            for name, net in mnets.items():
                # Ensure the network is in training mode
                net.net.train()

                if not step%4096:
                    result = torch.cat((net.bufferX, net.bufferY,net.bufferZ),1)
                    filename = name + '_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,step) + '.pt'
                    print(filename)
                    torch.save(result, os.path.join(info_dir, filename))

                avg_loss = net.update(X, y, mushroom_idx)
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
                    filename = name +'GPU' + '_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,step) + '.jpg'
                    print(filename)
                    plt.savefig(os.path.join(plot_dir, filename))

                    # Save the latest model
                    torch.save(net.net.state_dict(), os.path.join(net.net.model_save_dir, 'model.pt'))
                    

    # Save the cumulative regret for each network
    for name, net in mnets.items():
        np.save(os.path.join(net.net.model_save_dir, 'cum_regrets.npy'), np.array(net.cum_regrets))
        result = torch.cat((net.bufferX, net.bufferY,net.bufferZ),1)
        filename = name +'GPU'+ '_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,'final') + '.pt'
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
    filename = name +'GPU'+  '_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,'final') + '.jpg'
    plt.savefig(os.path.join(plot_dir, filename))

    df = pd.DataFrame(columns=['idx'],data=idx_list)
    filename = name +'GPU'+  '_lr_{0}_stepsize_{1}_gamma_{2}_epoch_{3}'.format(lr,step_size,gamma,'final')
    df.to_csv(filename+".csv", encoding='utf-8', index=False)
    # Show the plot
    # plt.show()


if __name__ == '__main__':
    run_rl_training()

