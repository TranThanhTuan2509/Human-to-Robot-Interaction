import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np
import collections
import random

LOG_STD_MAX = 0.0
LOG_STD_MIN = -5.0

def layer_init(layer, weight_gain=1, bias_const=0, args=None):
    if isinstance(layer, nn.Linear):
        if args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        elif args.weights_init == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if args.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

class Policy(nn.Module):
    def __init__(self, input_shape, out_c, out_d, args):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.mean = nn.Linear(256, out_c)
        self.logstd = nn.Linear(256, out_c)
        self.pi_d = nn.Linear(256, out_d)
        self.args = args
        self.apply(lambda layer: layer_init(layer, args=self.args))

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = torch.relu(self.fc1(x))
        mean = torch.tanh(self.mean(x))
        log_std = self.logstd(x)
        pi_d = self.pi_d(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std, pi_d

    def get_action(self, x, device):
        mean, log_std, pi_d = self.forward(x, device)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action_c = torch.tanh(x_t)
        all_log_prob_c = normal.log_prob(x_t)
        all_log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)
        log_prob_c = torch.cat([all_log_prob_c[:, :2].sum(1, keepdim=True), all_log_prob_c[:, 2:]], 1)
        dist = Categorical(logits=pi_d)
        action_d = dist.sample()
        prob_d = dist.probs
        log_prob_d = torch.log(prob_d + 1e-8)
        return action_c, action_d, log_prob_c, log_prob_d, prob_d

class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, out_c, out_d, args):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape + out_c, 256)
        self.fc2 = nn.Linear(256, out_d)
        self.args = args
        self.apply(lambda layer: layer_init(layer, args=self.args))

    def forward(self, x, a, device):
        x = torch.Tensor(x).to(device)
        a = torch.Tensor(a).to(device) if not isinstance(a, torch.Tensor) else a
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)
        return np.array(s_lst), np.array(a_lst), np.array(r_lst), np.array(s_prime_lst), np.array(done_mask_lst)