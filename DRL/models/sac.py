import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import shutil

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

def path_init(file_path):
    dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=512, fc2_dims=512, n_actions=None,
            name='critic', chkpt_dir='ckpt/sac', lr=None):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.fc3 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc4 = nn.Linear(fc1_dims, fc2_dims)
        self.q2 = nn.Linear(fc2_dims, 1)
        path_init(self.checkpoint_file)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.apply(weights_init_)

    def forward(self, state, action):
        # state = state.view(state.size(0), -1)
        # action = action.squeeze(1)
        # print(f"State shape: {state.shape}, Action shape: {action.shape}")
        combo = T.cat([state, action], dim=1)
        action = F.relu(self.fc1(combo))
        action = F.relu(self.fc2(action))
        q = self.q(action)

        action2 = F.relu(self.fc3(combo))
        action2 = F.relu(self.fc4(action2))
        q2 = self.q2(action2)

        return q, q2

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=512,
            fc2_dims=512, action_space=None, n_actions=2, name='actor', chkpt_dir='ckpt/sac', lr=None):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        path_init(self.checkpoint_file)
        self.apply(weights_init_)

        # action rescaling (based on max action)
        if action_space is None:
            self.action_scale = T.tensor(1.).to(self.device)
            self.action_bias = T.tensor(0.).to(self.device)
        else:
            self.action_scale = T.FloatTensor((action_space.high - action_space.low) / 2).to(self.device)
            self.action_bias = T.FloatTensor((action_space.high + action_space.low) / 2).to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mu, sigma

    def sample_normal(self, state, reparametrization=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma.exp())
        if reparametrization:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        action = T.tanh(actions) * self.action_scale + self.action_bias
        log_prob = probabilities.log_prob(actions)
        log_prob -= T.log(self.action_scale * (1 - action.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = T.tanh(mu) * self.action_scale + self.action_bias

        return action, log_prob, mu

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, input_dim, action_space, fc1_dim=512, fc2_dim=512, lr=0.0001,
                 target_update_interval=1, alpha=0.2, tau=0.005, discount=0.99, writer=None, a_lr=0.0001):

        self.alpha = alpha
        self.discount = discount
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.writer = writer
        self.updates = 0

        self.critic = CriticNetwork(input_dims=input_dim, n_actions=action_space.shape[0], fc1_dims=fc1_dim, fc2_dims=fc2_dim,
                             lr=lr, name=f'critic')
        self.critic_target = CriticNetwork(input_dims=input_dim, n_actions=action_space.shape[0], fc1_dims=fc1_dim,
                                    fc2_dims=fc2_dim, lr=lr, name=f'critic_target')
        self.actor = ActorNetwork(input_dims=input_dim, n_actions=action_space.shape[0], fc1_dims=fc1_dim, fc2_dims=fc2_dim, lr=lr,
                           action_space=action_space, name=f'actor')
        self.update_network_parameters(target=self.critic_target, source=self.critic, tau=1)

        # self.target_entropy = -T.prod(T.Tensor(action_space.shape[0]).to(self.actor.device)).item()
        self.target_entropy = -float(action_space.shape[0])
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

    def choose_action(self, observation, evaluate=False):
        state = T.FloatTensor(observation).to(self.actor.device).unsqueeze(0)
        if evaluate is False:
            actions, _, _ = self.actor.sample_normal(state, reparametrization=False)
        else:
            _, _, actions = self.actor.sample_normal(state, reparametrization=False)
        return actions.detach().cpu().numpy()[0]

    def learn(self, memory, batch_size):
        state_batch, next_state_batch, action_batch, reward_batch, mask_batch = memory.sample(batch_size)

        state_batch = T.FloatTensor(state_batch).to(self.actor.device)
        action_batch = T.FloatTensor(action_batch).to(self.actor.device)
        next_state_batch = T.FloatTensor(next_state_batch).to(self.actor.device)
        reward_batch = T.FloatTensor(reward_batch).to(self.actor.device).unsqueeze(1)
        mask_batch = T.FloatTensor(mask_batch).to(self.actor.device).unsqueeze(1)

        # compute critic loss
        with T.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample_normal(next_state_batch,
                                                                               reparametrization=False)
            qf1_next_target, qf2_next_target = self.critic_target.forward(next_state_batch, next_state_action)
            min_qf_next_target = T.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.discount * (min_qf_next_target)

        qf1, qf2 = self.critic.forward(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        self.writer.add_scalar(
            "Q/Q1", qf1_loss.item(), self.updates
        )
        qf2_loss = F.mse_loss(qf2, next_q_value)
        self.writer.add_scalar(
            "Q/Q2", qf2_loss.item(), self.updates
        )
        qf_loss = qf1_loss + qf2_loss

        self.critic.optimizer.zero_grad()
        qf_loss.backward()
        self.critic.optimizer.step()

        # compute policy loss
        pi, log_pi, _ = self.actor.sample_normal(state_batch, reparametrization=True)
        qf1_pi, qf2_pi = self.critic.forward(state_batch, pi)
        min_qf_pi = T.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        # Update tempature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()
        alpha_tlogs = T.tensor(self.alpha)
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        # T.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.writer.add_scalar(
            "Loss/policy_loss", policy_loss.item(), self.updates
        )

        self.writer.add_scalar(
            "Loss/alpha_loss", alpha_loss.item(), self.updates
        )

        self.writer.add_scalar(
            "Loss/alpha_tlogs", alpha_tlogs.item(), self.updates
        )

        if self.updates % self.target_update_interval == 0:
            # define soft update function
            self.update_network_parameters(target=self.critic_target, source=self.critic)
            
        self.updates += 1
        
    def update_network_parameters(self, target, source, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()