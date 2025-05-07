import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.RelayBuffer import ReplayBuffer
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.isdir("./OurTD3_loss"):
    shutil.rmtree("./OurTD3_loss")
os.makedirs("./OurTD3_loss")

writer = SummaryWriter("./OurTD3_loss")


# Implementation of *Modified* Twin Delayed Deep Deterministic Policy Gradients (TD3)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=6, p_action_dim=5, suction_dim=2, max_action=6):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)  # robot joint
        self.p_action = nn.Linear(300, p_action_dim)  # possible action
        self.suction = nn.Linear(300, suction_dim)  # suction state

        self.max_action = max_action

        # self.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        p_action_logits = self.p_action(x)  # possible action, raw outputs (a.k.a. logits) from the nn.Linear layers
        suction_logits = self.suction(x)  # suction state, raw outputs (a.k.a. logits) from the nn.Linear layers
        x = self.max_action * torch.tanh(self.l3(x))  # robot joint

        return x, p_action_logits, suction_logits  # shape: [batch_size, 6], [batch_size, 5] raw outputs, [batch_size, 2] raw outputs

    def sample_discrete(self, action_logits, suction_logits):
        # Sample discrete actions (argmax for simplicity; could use Gumbel-Softmax)
        action_probs = F.softmax(action_logits, dim=-1)
        suction_probs = F.softmax(suction_logits, dim=-1)
        action_id = torch.argmax(action_probs, dim=-1)
        suction_id = torch.argmax(suction_probs, dim=-1)
        return action_id, suction_id  # shape: [batch_size, 1], [batch_size, 1]


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=6, p_action_dim=5, suction_dim=2):
        super(Critic, self).__init__()
        # TD3 uses 2 critic networks to evaluate the Q_value as the good of the taken action at the chosen state

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + p_action_dim + suction_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + p_action_dim + suction_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        # Q3 architecture
        self.l7 = nn.Linear(state_dim + action_dim + p_action_dim + suction_dim, 400)
        self.l8 = nn.Linear(400, 300)
        self.l9 = nn.Linear(300, 1)

        # self.apply(weights_init)

    def forward(self, x, u, p, s):  # state, action, possible_action, suction
        xu = torch.cat([x, u, p, s], -1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        x3 = F.relu(self.l7(xu))
        x3 = F.relu(self.l8(x3))
        x3 = self.l9(x3)
        return x1, x2, x3  # shape: [batch_size, 1], [batch_size, 1], [batch_size, 1]

    def Q1(self, x, u, p, s):
        # Test Critic Q1
        xu = torch.cat([x, u, p, s], -1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim, p_action_dim, suction_dim, max_action):
        self.actor = Actor(state_dim, action_dim, p_action_dim, suction_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, p_action_dim, suction_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.update_step = 0
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.p_action_dim = p_action_dim
        self.suction_dim = suction_dim

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        actions, p_action_logits, suction_logits = self.actor(state)
        p_action_id, suction_id = self.actor.sample_discrete(p_action_logits, suction_logits)

        return actions.cpu().data.numpy().flatten(), \
               p_action_id.cpu().data.numpy().flatten(), \
               suction_id.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        if len(replay_buffer.storage) < batch_size:
            return

        # Sample replay buffer
        ids, states, next_states, actions, rewards, dones = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(states).to(device)  # Shape: [batch_size, 524]

        action, p_action, suction = actions
        action = torch.FloatTensor(action).to(device)
        p_action = torch.LongTensor(p_action).to(device)
        suction = torch.LongTensor(suction).to(device)

        next_state = torch.FloatTensor(next_states).to(device)
        done = torch.FloatTensor(1 - dones).to(device)
        reward = torch.FloatTensor(rewards).to(device)

        # Convert discrete IDs to one-hot
        p_action_onehot = F.one_hot(p_action, num_classes=self.p_action_dim).squeeze(
            1).float()  # shape: [batch_size, 5]
        suction_onehot = F.one_hot(suction, num_classes=self.suction_dim).squeeze(1).float()  # shape: [batch_size, 2]

        # Select action according to policy and add clipped noise
        # Adding noise for exploring
        noise = torch.randn_like(action, device=action.device).normal_(0, policy_noise)
        noise = noise.clamp(-noise_clip, noise_clip)

        next_action, next_p_action, next_suction = self.actor_target(next_state)
        next_p_action_id, next_suction_id = self.actor_target.sample_discrete(next_p_action,
                                                                              next_suction)  # shape: [batch_size,1], [batch_size,1]

        # Next action = noise + actor target of next state
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)  # shape: [batch_size, 6]
        # One hot encoder for both pa and su
        next_p_action_onehot = F.one_hot(next_p_action_id,
                                         num_classes=self.p_action_dim).float()  # shape: [batch_size, 5]
        next_suction_onehot = F.one_hot(next_suction_id, num_classes=self.suction_dim).float()  # shape: [batch_size, 2]

        # Compute the target Q-value
        target_Q1, target_Q2, target_Q3 = self.critic_target(next_state, next_action, next_p_action_onehot,
                                                             next_suction_onehot)
        target_Qs = torch.stack([target_Q1, target_Q2, target_Q3], dim=1)  # [batch_size, 3]
        median_target_Q, _ = torch.median(target_Qs, dim=1)  # [batch_size]
        target_Q = reward + (done * discount * median_target_Q).detach()

        # Get current Q estimates as approximated Q-value
        current_Q1, current_Q2, current_Q3 = self.critic(state, action, p_action_onehot, suction_onehot)

        # Compute critic loss/ TD error/ MSE loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q) + \
                      F.mse_loss(current_Q3, target_Q)
        writer.add_scalar('critic loss', critic_loss, self.update_step)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        # Critic backpropagation
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.update_step % policy_freq == 0:
            # Compute actor loss by taking the median between the three critic networks,
            # avoiding underestimate and overestimate
            actions, p_action_logits, suction_logits = self.actor(state)
            p_action_probs = F.softmax(p_action_logits, dim=-1)
            suction_probs = F.softmax(suction_logits, dim=-1)
            q_values = []
            for p_a in range(self.p_action_dim):
                for s in range(self.suction_dim):
                    p_a_onehot = F.one_hot(torch.tensor(p_a, device=device),
                                           num_classes=self.p_action_dim).float().expand(batch_size, -1)
                    s_onehot = F.one_hot(torch.tensor(s, device=device),
                                         num_classes=self.suction_dim).float().expand(batch_size, -1)
                    q1, q2, q3 = self.critic(state, actions, p_a_onehot, s_onehot)
                    qs = torch.stack([q1, q2, q3], dim=1)  # [batch_size, 3]
                    median_q, _ = torch.median(qs, dim=1)  # [batch_size]
                    median_q = median_q.unsqueeze(1)  # [batch_size, 1]
                    weighted_q = median_q * p_action_probs[:, p_a:p_a + 1] * suction_probs[:,
                                                                             s:s + 1]  # [batch_size, 1]
                    q_values.append(weighted_q)  # end shape: [batch_size, p_a*s]
            # DPG error
            actor_loss = -torch.sum(torch.cat(q_values, dim=1), dim=1).mean()
            writer.add_scalar('actor loss', actor_loss, self.update_step)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            # Actor backpropagation
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        self.update_step += 1

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))