from env.robot_env import ImitationLearning
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from draft3 import DQNAgent, ReplayMemory
import torch
from models.text_encoder import Encoder
from models.TransporterNet import TransporterNetsEval
from config import *

env = ImitationLearning()
n_states = 512
n_actions = env.action_space.n
agent = DQNAgent(alpha=0.0001, n_states=n_states, n_actions=n_actions)
encoder = Encoder(cpu=False)
memory = ReplayMemory()
transporterNet_checkpoint_path = '...'
model = TransporterNetsEval(transporterNet_checkpoint_path)
writer = SummaryWriter('./log/dqn_trial_1')
load_models = False
n_episodes = 2000
n_steps = 300
MIN_REPLAY_SIZE = 100
# Load weights
if load_models:
    agent.load_models()

total_reward_hist = []
avg_reward_hist = []
# Initialize Replay Buffer
for _ in range(MIN_REPLAY_SIZE):
    state = env.reset()
    action = agent.epsilon_greedy(state)
    ids, next_state, reward, done, _ = env.step(action)

    agent.store_transition(state, action, reward, next_state, done)
    '''assigns new env'''

# Training Loop
for episode in range(1, n_episodes + 1):
    # state = [obs, instruction]
    ids, state, action, reward, next_state, done = memory.sample_memory(1)
    total_reward = 0

    # for t in range(n_steps):
    while not done:
        # Uncomment to render after episode 1800
        # if episode > 1800:
        #     env.render()
        action = agent.epsilon_greedy(state)
        action = torch.hstack(action, ids)
        ids, next_state, reward, done, _ = env.step(action)  # Use instruction
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        # if done:
        #     break

    if not load_models:
        agent.decrement_epsilon()

    # Save model periodically
    if episode > 1000 and episode % 200 == 0:
        agent.save_models()

    total_reward_hist.append(total_reward)
    avg_reward = np.average(total_reward_hist[-100:])
    avg_reward_hist.append(avg_reward)
    print("Episode :", episode, "Epsilon : {:.2f}".format(agent.eps), "Total Reward : {:.2f}".format(total_reward),
          "Avg Reward : {:.2f}".format(avg_reward))
    # Tensorboard log
    writer.add_scalar('Total Reward', total_reward, episode)
    writer.add_scalar('Avg Reward', avg_reward, episode)
    writer.add_scalar('Epsilon', agent.eps, episode)
    
fig, ax = plt.subplots()
t = np.arange(n_episodes)
ax.plot(t, total_reward_hist, label="Total Reward")
ax.plot(t, avg_reward_hist, label="Average Reward")
ax.set_title("Reward vs Episode")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.legend()
plt.show()