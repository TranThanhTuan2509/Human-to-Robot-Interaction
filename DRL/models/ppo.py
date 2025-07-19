import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import shutil

def path_init(file_path):
    dir_path = os.path.dirname(file_path)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOMemory:
    def __init__(self):
        self.states = []
        self.privileges = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        # self.batch_size = batch_size
    '''for better learning i should collect data over a fixed number of steps and update the networks only after collecting all'''
    def generate_batches(self):
        return (np.array(self.states), np.array(self.privileges), np.array(self.actions), np.array(self.probs),
                np.array(self.vals), np.array(self.rewards), np.array(self.dones))

    def store_memory(self, state, privilege, action, probs, vals, reward, done, true_done):
        self.states.append(state)
        self.privileges.append(privilege)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.privileges = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-08):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.ones(()))

        self.epsilon = epsilon

    def forward(self, obs, update=True):
        if update:
            self.update(obs)

        return (obs - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, correction=0, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.running_mean, self.running_var, self.count = (
            update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                batch_mean,
                batch_var,
                batch_count,
            )
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, chkpt_dir='tmp/ppo', name='actor'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ppo')
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(input_dims).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, np.prod(n_actions)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(n_actions)))
        path_init(self.checkpoint_file)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        action_mean = self.actor(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, chkpt_dir='tmp/ppo', name='critic'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ppo')
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(input_dims).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        path_init(self.checkpoint_file)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, alpha=0.0003, ppo_cfg=None, writer=None):
        self.alpha = alpha
        self.actor = ActorNetwork(n_actions, input_dims, self.alpha)
        self.critic = CriticNetwork(input_dims + 14, self.alpha)
        self.memory = PPOMemory()
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(n_actions)))

        self.GAMMA = ppo_cfg.gamma
        self.POLICY_CLIP = ppo_cfg.policy_clip
        self.GAE_LAMBDA = ppo_cfg.gae_lambda
        self.UPDATES_EPOCHS = ppo_cfg.updates_epochs
        self.BATCH_SIZE = ppo_cfg.batch_size
        self.MINIBATCH_SIZE = ppo_cfg.minibatch_size
        self.VF_COEF = ppo_cfg.vf_coef
        self.ENT_COEF = ppo_cfg.ent_coef
        self.CLIP_COEF = ppo_cfg.clip_coef

        self.obs_rms = RunningMeanStd(shape=input_dims).to(self.actor.device)
        self.cmb_rms = RunningMeanStd(shape=input_dims+14).to(self.actor.device)
        self.value_rms = RunningMeanStd(shape=()).to(self.actor.device)
        self.writer = writer
        path_init()

    def remember(self, state, privilege, action, probs, vals, reward, done, true_done):
        self.memory.store_memory(state, privilege, action, probs, vals, reward, done, true_done)

    def save_models(self):
        print('saving...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('loading...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        normal = observation['normal']
        privileged = observation['privileged']
        state = torch.tensor([normal], dtype=torch.float).to(self.actor.device)
        privileged = torch.tensor([privileged], dtype=torch.float).to(self.actor.device)

        privileged = torch.cat([state, privileged], -1)
        cmb = self.cmb_rms.forward(privileged)
        state = self.obs_rms.forward(state)

        dist = self.actor.forward(state)
        value = self.critic.forward(cmb)
        action = dist.sample()
        probs = dist.log_prob(action).sum(1).squeeze().item()
        action = action.squeeze().cpu().numpy()  # Convert to numpy for env.step
        value = value.squeeze().item()
        print('\n', action)
        return action, probs, value

    def learn(self, iteration, done, true_done):

        # Get data from memory
        state_arr, privilege_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr = self.memory.generate_batches()

        # Convert to tensors
        state_tensor = torch.tensor(state_arr, dtype=torch.float).to(self.actor.device)
        privilege_tensor = torch.tensor(privilege_arr, dtype=torch.float).to(self.actor.device)
        action_tensor = torch.tensor(action_arr, dtype=torch.float).to(self.actor.device)
        old_probs_tensor = torch.tensor(old_probs_arr, dtype=torch.float).to(self.actor.device)
        vals_tensor = torch.tensor(vals_arr, dtype=torch.float).to(self.actor.device)
        reward_tensor = torch.tensor(reward_arr, dtype=torch.float).to(self.actor.device)
        done_tensor = torch.tensor(done_arr, dtype=torch.float).to(self.actor.device)

        # Compute advantages using GAE
        N = len(reward_arr)
        # not_dones = 1.0 - done_tensor  # 1 - deltat (probability of termination)
        # reward_tensor *= not_dones  # if (1 - deltat) towards 1 as non-terminated then the rewards are un-changed

        advantages = torch.zeros(N, device=self.actor.device)
        gae = 0
        for t in reversed(range(N)):
            next_val = vals_tensor[t+1]

            delta = reward_tensor[t] + self.GAMMA * next_val - vals_tensor[t] #rt + V(t+1) - V(t)
            gae = delta + self.GAMMA * self.GAE_LAMBDA * gae #delta(t) + gamma*lambda*delta(t-1) +...
            advantages[t] = gae

        returns_batch = advantages + vals_tensor

        returns_batch = self.value_rms.forward(returns_batch)
        vals_tensor = self.value_rms.forward(vals_tensor)

        sum_pg_loss = sum_entropy_loss = sum_v_loss = sum_surrogate_loss = 0.0
        # For each epoch
        for epoch in range(self.UPDATES_EPOCHS):
            b_inds = torch.randperm(self.BATCH_SIZE, device=self.actor.device)
            for start in range(0, self.BATCH_SIZE, self.MINIBATCH_SIZE):
                end = start + self.MINIBATCH_SIZE
                mb_inds = b_inds[start:end]
                states = state_tensor[mb_inds]
                privileges = privilege_tensor[mb_inds]
                actions = action_tensor[mb_inds]
                old_probs = old_probs_tensor[mb_inds]
                adv_batch = advantages[mb_inds]

                # Compute current policy and value
                privileged = torch.cat([states, privileges], -1)
                cmb = self.cmb_rms.forward(privileged)
                states = self.obs_rms.forward(states)

                dist = self.actor.forward(states)
                critic_value = self.critic.forward(cmb).squeeze()
                critic_value = self.value_rms.forward(critic_value, update=False)

                # Compute policy loss
                new_probs = dist.log_prob(actions).sum(1)
                logratio = new_probs - old_probs
                prob_ratio = logratio.exp()

                adv_batch = (adv_batch - adv_batch.mean()) / (
                        adv_batch.std() + 1e-8
                )
                surr1 = -adv_batch * prob_ratio
                surr2 = -adv_batch * torch.clamp(prob_ratio, 1 - self.POLICY_CLIP, 1 + self.POLICY_CLIP)
                actor_loss = torch.max(surr1, surr2).mean()

                # Compute value loss
                v_loss_unclipped = (returns_batch[mb_inds] - critic_value) ** 2

                v_clipped = vals_tensor[mb_inds] + torch.clamp(
                    critic_value - vals_tensor[mb_inds],
                    -self.CLIP_COEF,
                    self.CLIP_COEF,
                )
                v_loss_clipped = (v_clipped - returns_batch[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Compute entropy
                entropy = dist.entropy().sum(1).mean()

                # Total loss with entropy bonus
                total_loss = actor_loss + v_loss * self.VF_COEF - self.ENT_COEF * entropy

                sum_pg_loss += actor_loss
                sum_entropy_loss += entropy
                sum_v_loss += v_loss
                sum_surrogate_loss += total_loss

                # Optimize
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        num_updates = self.UPDATES_EPOCHS * self.BATCH_SIZE / self.MINIBATCH_SIZE
        self.writer.add_scalar(
            "Loss/mean_pg_loss", sum_pg_loss / num_updates, iteration
        )
        self.writer.add_scalar(
            "Loss/mean_entropy_loss", sum_entropy_loss / num_updates, iteration
        )
        self.writer.add_scalar("Loss/mean_v_loss", sum_v_loss / num_updates, iteration)
        self.writer.add_scalar(
            "Loss/mean_surrogate_loss", sum_surrogate_loss / num_updates, iteration
        )

        self.memory.clear_memory()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')
        print('saved')

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/{filename}_critic.pth'))
        print('loaded')

