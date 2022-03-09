# working directory
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from bbb.utils.pytorch_setup import DEVICE
from bbb.config.parameters import Parameters, PriorParameters
from bbb.config.constants import KL_REWEIGHTING_TYPES, PRIOR_TYPES, VP_VARIANCE_TYPES
from bbb.models.dnn import DNN
import torch.optim as optim
from bbb.data import load_bandit
# importing necessary modules
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from bbb.models.bnn import RegressionBNN

logger = logging.getLogger(__name__)

X,y = load_bandit("/Users/yufanwang/Desktop/Study/AML/BBB/src/bbb/mushroom.csv")


# coding a class for this
class MushroomBandit:
    # initializing
    def __init__(self, X, y,n_weight_sampling=2):
        self.epsilon = 0
        self.net = None
        self.loss, self.optimizer = None, None
        self.bufferX, self.bufferY = [], []
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

    def init_buffer(self):
        for i in np.random.choice(range(len(X)), 4096):
            eaten = 1 if np.random.rand() > 0.5 else 0
            self.bufferX.append(X[i])
            self.bufferY.append(self.get_reward(eaten, y[i]))
        
    # function to get which mushrooms will be eaten
    def eat_mushrooms(self, mushroom_idx):
        context, poison = X[mushroom_idx], y[mushroom_idx]
        tensor_context = torch.from_numpy(context).type(torch.FloatTensor)

        if np.random.rand() < self.epsilon:
            eaten = int(np.random.rand() < 0.5)
        else:
            with torch.no_grad():
                predict_reward = sum([self.net(tensor_context) for _ in range(self.n_weight_sampling)]).item()
                eaten = 0 if predict_reward > 0 else 1
        
        agent_reward = self.get_reward(eaten, poison)

        # Get rewards and update buffer
        self.bufferX.append(context)
        self.bufferY.append(agent_reward)

        # Calculate regret
        regret = self.calculate_regret(agent_reward,poison)
        self.cum_regrets.append(self.cum_regrets[-1]+regret)

    # Update buffer
    def update(self, mushroom):
        self.eat_mushrooms(mushroom)
        # idx pool
        l = len(self.bufferX)
        idx_pool = range(l) if l >= 4096 else ((int(4096//l) + 1)*list(range(l)))
        idx_pool = np.random.permutation(idx_pool[-4096:])
        context_pool = torch.Tensor([self.bufferX[i] for i in idx_pool])
        reward_pool = torch.Tensor([self.bufferY[i] for i in idx_pool])
        for i in range(0, 4096, 64):
            self.loss_step(context_pool[i:i+64], reward_pool[i:i+64], i//64)
        
    
    def loss_step(self, x, y):
        raise NotImplementedError

DNN_REGRESSION_PARAMETERS = Parameters(
    name = "DNN_regression",
    input_dim = X.shape[1],
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
        self.mse = lambda x, y:((x-y)**2).sum()/64
        
    def loss_step(self, x, y, batch_id):
        self.net.zero_grad()
        loss = self.mse(self.net.forward(x), y)
        loss.backward()
        self.optimizer.step()

BNN_REGRESSION_PARAMETERS = Parameters(
    name = "BBB_regression",
    input_dim = X.shape[1],
    output_dim = 1,
    weight_mu = [-0.2, 0.2],
    weight_rho = [-5, -4],
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
        self.net = RegressionBNN(params=BNN_REGRESSION_PARAMETERS).to(DEVICE)
        self.n_samples = BNN_REGRESSION_PARAMETERS.name
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        
    def loss_step(self, x, y, batch_id):
        # print(batch_id)
        beta = 2 ** (64 - (batch_id + 1)) / (2 ** 64 - 1) 
        self.net.model.train()
        self.net.zero_grad()
        # print(X.shape[1])
        loss = self.net.sample_ELBO(x, y, beta, 2)
        # print(loss)
        net_loss = loss[0]
        net_loss.backward()
        self.optimizer.step()

X = X.to_numpy()
y = y.to_numpy()

rate=1e-5
mnets = {'Greedy':Greedy(X=X,y=y,lr=rate,epsilon=0),
         'Greedy 1%':Greedy(X=X,y=y,lr=rate, epsilon=0.01),
         'Greedy 5%':Greedy(X=X,y=y,lr=rate, epsilon=0.05)}

NB_STEPS = 100
torch.autograd.set_detect_anomaly(True)

for name, net in mnets.items():
    net.init_buffer()
for step in tqdm(range(NB_STEPS)):
    mushroom = np.random.randint(len(X))
    for name, net in mnets.items():
        net.update(mushroom)

fig, ax = plt.subplots() 
for name, net in mnets.items():
    ax.plot(net.cum_regrets, label=name)
ax.set_xlabel('Steps') 
ax.set_ylabel('Regret') 
ax.legend()
plt.show()

