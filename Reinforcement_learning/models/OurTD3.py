import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.RelayBuffer import ReplayBuffer
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize TensorBoard logging
if os.path.isdir("./OurTD3_loss"):
    shutil.rmtree("./OurTD3_loss")
os.makedirs("./OurTD3_loss")
writer = SummaryWriter("./OurTD3_loss")

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=6, p_action_dim=5, max_action=6):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.p_action = nn.Linear(300, p_action_dim)
        self.suction = nn.Linear(300, 1)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        joint_action = self.max_action * torch.tanh(self.l3(x))
        p_action_logits = self.p_action(x) # possible action, raw outputs (a.k.a. logits) from the nn.Linear layers
        suction_logit = self.suction(x) # suction state, raw outputs (a.k.a. logits) from the nn.Linear layers
        return joint_action, p_action_logits, suction_logit # shape: [batch_size, 6], [batch_size, 5] raw outputs, [batch_size, 2] raw outputs

    def sample_discrete(self, p_action_logits, suction_logit):
        p_action_probs = F.softmax(p_action_logits, dim=-1)
        suction_prob = torch.sigmoid(suction_logit)
        p_action_id = torch.argmax(p_action_probs, dim=-1, keepdim=True)
        suction_id = (suction_prob > 0.5).long()
        return p_action_id, suction_id # shape: [batch_size, 1], [batch_size, 1]

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=6, p_action_dim=5):
        super(Critic, self).__init__()
        self.p_action_embed = nn.Embedding(p_action_dim, 16)
        input_dim = state_dim + action_dim + 16 + 1
        self.l1 = nn.Linear(input_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        self.l4 = nn.Linear(input_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)
        self.l7 = nn.Linear(input_dim, 400)
        self.l8 = nn.Linear(400, 300)
        self.l9 = nn.Linear(300, 1)
        self.apply(weights_init)

    def forward(self, x, u, p_action_id, suction):
        suction = suction.view(-1, 1)
        p_action_embed = self.p_action_embed(p_action_id).squeeze(1)
        xu = torch.cat([x, u, p_action_embed, suction], -1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        
        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        x3 = F.relu(self.l7(xu))
        x3 = F.relu(self.l8(x3))
        x3 = self.l9(x3)
        return x1, x2, x3 # shape: [batch_size, 1], [batch_size, 1], [batch_size, 1]

    def Q1(self, x, u, p_action_id, suction):
        p_action_embed = self.p_action_embed(p_action_id).squeeze(1)
        xu = torch.cat([x, u, p_action_embed, suction], -1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

class TD3:
    def __init__(self, state_dim, action_dim, p_action_dim, max_action, actor_lr=1e-4, critic_lr=1e-3):
        self.actor = Actor(state_dim, action_dim, p_action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, p_action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(state_dim, action_dim, p_action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, p_action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.max_action = max_action
        self.p_action_dim = p_action_dim
        self.update_step = 0
        self.state_mean = 0.0
        self.state_std = 1.0
        self.normalizer_initialized = False

    def update_normalizer(self, batch_states):
        if not self.normalizer_initialized and self.update_step < 1000:
            self.state_mean = np.mean(batch_states, axis=0)
            self.state_std = np.std(batch_states, axis=0) + 1e-6
            if self.update_step == 999:
                self.normalizer_initialized = True
        else:
            new_mean = np.mean(batch_states, axis=0)
            new_std = np.std(batch_states, axis=0) + 1e-6
            self.state_mean = 0.95 * self.state_mean + 0.05 * new_mean
            self.state_std = 0.95 * self.state_std + 0.05 * new_std

    def select_action(self, state):
        state = (state - self.state_mean) / self.state_std
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        joint_action, p_action_logits, suction_logit = self.actor(state)
        p_action_id, suction_id = self.actor.sample_discrete(p_action_logits, suction_logit)
        return (joint_action.cpu().data.numpy().flatten(),
                p_action_id.cpu().data.numpy().flatten(),
                suction_id.cpu().data.numpy().flatten())

    def train(self, replay_buffer, batch_size=128, discount=0.99, tau=0.001, policy_noise=0.1, noise_clip=0.2, policy_freq=2):
        if len(replay_buffer.storage) < batch_size:
            return

        ids, states, next_states, actions, rewards, dones = replay_buffer.sample(batch_size)
        self.update_normalizer(states)
        states = (states - self.state_mean) / self.state_std
        next_states = (next_states - self.state_mean) / self.state_std

        state = torch.FloatTensor(states).to(device)
        next_state = torch.FloatTensor(next_states).to(device)
        action, p_action, suction = actions
        action = torch.FloatTensor(action).to(device)
        p_action = torch.LongTensor(p_action).to(device)
        suction = torch.FloatTensor(suction).to(device).unsqueeze(-1)
        done = torch.FloatTensor(1 - dones).to(device)
        reward = torch.FloatTensor(rewards).to(device)

        # writer.add_scalar('Reward/mean', reward.mean().item(), self.update_step)
        # writer.add_scalar('Reward/min', reward.min().item(), self.update_step)
        # writer.add_scalar('Reward/max', reward.max().item(), self.update_step)
        # writer.add_scalar('State/mean_norm', self.state_mean.mean(), self.update_step)
        # writer.add_scalar('State/std_norm', self.state_std.mean(), self.update_step)

        # Next action = noise + actor target of next state
        noise = torch.randn_like(action).normal_(0, policy_noise).clamp(-noise_clip, noise_clip)
        next_action, next_p_action_logits, next_suction_logit = self.actor_target(next_state)
        next_p_action_id, next_suction_id = self.actor_target.sample_discrete(next_p_action_logits, next_suction_logit)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
        next_suction = next_suction_id.float().unsqueeze(-1)

        # Compute the target Q-value
        target_Q1, target_Q2, target_Q3 = self.critic_target(next_state, next_action, next_p_action_id, next_suction)
        target_Qs = torch.stack([target_Q1, target_Q2, target_Q3], dim=1)
        median_target_Q, _ = torch.median(target_Qs, dim=1)
        target_Q = reward + (done * discount * median_target_Q).detach()
        # writer.add_scalar('Q/median_target', median_target_Q.mean().item(), self.update_step)

        current_Q1, current_Q2, current_Q3 = self.critic(state, action, p_action, suction)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + F.mse_loss(current_Q3, target_Q)
        writer.add_scalar('Loss/Critic', critic_loss.item(), self.update_step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        # writer.add_scalar('Grad/critic_norm', critic_grad_norm, self.update_step)
        self.critic_optimizer.step()

        if self.update_step % policy_freq == 0:
            joint_action, p_action_logits, suction_logit = self.actor(state)
            p_action_probs = F.softmax(p_action_logits, dim=-1)
            suction_prob = torch.sigmoid(suction_logit).unsqueeze(-1)
            suction_probs = torch.cat([1 - suction_prob, suction_prob], dim=-1)  # [batch_size, 2]

            q_values = []
            for p_a in range(self.p_action_dim):
                for s in range(2):  # suction_dim=2
                    p_a_id = torch.tensor(p_a, device=device).expand(batch_size)
                    s_id = torch.tensor(s, device=device).expand(batch_size).float().unsqueeze(-1)
                    q1, q2, q3 = self.critic(state, joint_action, p_a_id, s_id)
                    qs = torch.stack([q1, q2, q3], dim=1)
                    median_q, _ = torch.median(qs, dim=1)
                    weighted_q = median_q * p_action_probs[:, p_a] * suction_probs[:, s]
                    q_values.append(weighted_q)
            actor_loss = -torch.stack(q_values, dim=1).sum(dim=1).mean()
            writer.add_scalar('Loss/Actor', actor_loss.item(), self.update_step)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
            # writer.add_scalar('Grad/actor_norm', actor_grad_norm, self.update_step)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        # Update the frozen target models
        writer.add_scalar('Q/Q1_mean', current_Q1.mean().item(), self.update_step)
        writer.add_scalar('Q/Q2_mean', current_Q2.mean().item(), self.update_step)
        writer.add_scalar('Q/Q3_mean', current_Q3.mean().item(), self.update_step)
        self.update_step += 1

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/{filename}_critic.pth'))