import numpy as np
import torch
from datetime import datetime
import gym
import env
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time
from utils.RelayBuffer import ReplayBuffer
import matplotlib.pyplot as plt
import random
from moviepy.editor import ImageSequenceClip
from models.sac import Agent
import warnings
warnings.filterwarnings("ignore")

def evaluate(policy, eval_episodes=3):
    rewards = []
    for _ in range(eval_episodes):
        state, total, done = env.reset(), 0.0, False
        while not done:
            action = policy.choose_action(state)
            obs, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
    return np.mean(rewards)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="SAC")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval_freq", default=4e4, type=int)  # reward log path
    parser.add_argument("--log_path", default='./sac/reward_log', type=str)  # reward log path
    parser.add_argument("--record_path", default='./sac/record', type=str)  # reward log path
    parser.add_argument("--max_episode", default=1e5, type=float)  # Max episode to run environment for
    parser.add_argument("--exit_step", default=1000, type=float)  # Max episode to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic (recommend 128)
    parser.add_argument("--record_episode", default=30000, type=int)  # record episode
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--max_steps", default=100, type=int)  # max steps per eposide
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    args = parser.parse_args()
    print("---------------------------------------")
    print("*WARN*: Have you activated checkpoint saving mode yet?")
    print("---------------------------------------")

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    # Initialize saving file
    for dir_path in [args.log_path, args.record_path]:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # if os.path.isfile("./reward.txt"):
    #     os.remove("./reward.txt")
    # with open('./reward.txt', 'w') as f:
    #     pass

    writer = SummaryWriter(args.log_path)
    env = gym.make(args.env_name)
    env.unwrapped.max_steps = args.max_steps #define max steps
    # Set seeds
    random.seed(args.seed)
    env.unwrapped.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True #deterministic cuda
    torch.backends.cudnn.benchmark = False

    state_dim = env.observation_space.shape[0] # 535
    action_dim = env.action_space.shape[0] # 7 joints
    max_action = float(env.action_space.high[0])
    p_action = env.p_action_space.n # 5 possible actions

    # Initialize policy
    policy = Agent(input_dim=state_dim, action_space=env.action_space, writer=writer,
                 discount=args.discount, tau=args.tau)
    replay_buffer = ReplayBuffer()

    curr_episode = 0
    env.unwrapped.curriculum_learning = curr_episode + 40000
    best_eval_reward = float('-inf')
    early_stopping = -float('inf')
    timesteps_since_eval = 0
    updates_per_step = 4
    env.unwrapped.action_type = 3
    eval_history = []
    episode_reward = 0
    reward_history = []
    success = []
    t0 = time.time()
    '''TRAINING PROCESS'''
    while curr_episode < args.max_episode:
        total_reward = 0
        episode_timesteps = 0
        state = env.reset()
        done = False

        while not done:
            # Use policy with exploration noise
            action = policy.choose_action(state)
            if replay_buffer.ready(batch_size=args.batch_size):
                for i in range(updates_per_step):
                    policy.learn(replay_buffer, args.batch_size)

            next_state, reward, done, info = env.step(action)
            total_reward += reward #total step reward
            replay_buffer.add((state, next_state, action, reward, done))
            state = next_state

            episode_timesteps += 1
            reward_history.append(total_reward)

            print('------------------------------------------------------')
            print(f"Episode: {curr_episode} Step: {episode_timesteps} "
                f"Reward: {reward:.5f}  --  Wallclk T: {int(time.time() - t0)} sec")
            print('------------------------------------------------------')

        if early_stopping <= total_reward / episode_timesteps and total_reward / episode_timesteps > 1.5:
            env.unwrapped.record = True
            early_stopping = total_reward / episode_timesteps
            mean_r = evaluate(policy=policy)
            eval_history.append(mean_r)
            writer.add_scalar(f"Action_{env.unwrapped.action_type}/eval_reward", mean_r, curr_episode)

            # Save best model
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r
                policy.save()
                if env.cache_video:
                    # Create unique filename using timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = f"{args.record_path}/debug_{timestamp}_episode{curr_episode}_action_{env.unwrapped.action_type}.mp4"
                    debug_clip = ImageSequenceClip(env.cache_video, fps=60)
                    debug_clip.write_videofile(video_path, codec="libx264")


        # env.unwrapped.action_type += 1
        # early_stopping = -float('inf')
        # best_eval_reward = float('-inf')

        episode_reward += total_reward
        writer.add_scalar('Reward/avg reward', np.mean(reward_history[-100:]), curr_episode)
        writer.add_scalar('Reward/episode reward', episode_reward, curr_episode)
        writer.add_scalar('Reward/per episode reward', total_reward, curr_episode)
        # saving reward during training
        # with open("./reward.txt", 'a') as file:
        #     file.write(f"{episode_reward}\n")
        curr_episode += 1
        timesteps_since_eval += 1
        print('------------------------------------------------------')
        print(f"Episode: {curr_episode} Reward: {total_reward:.5f}  --  Wallclk T: {int(time.time() - t0)} sec")
        print('------------------------------------------------------')
