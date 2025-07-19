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
import random
from moviepy.editor import ImageSequenceClip
from models.TD3 import Agent
import warnings
warnings.filterwarnings("ignore")

def evaluate(policy, eval_episodes=1):
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
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--warmup", default=20000,
                        type=int)  # How many time steps purely random policy is run for (20000 for all actions)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_path", default='./td3/reward_log', type=str)  # reward log path
    parser.add_argument("--record_path", default='./td3/record', type=str)  # record log path
    parser.add_argument("--max_episode", default=1e6, type=float)  # Max episode to run environment for
    parser.add_argument("--expl_noise", default=0.2, type=float)  # Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic (recommend 128)
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--max_steps", default=100, type=int)  # max steps per episode
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    # Initialize saving file
    for dir_path in [args.log_path, args.record_path]:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    writer = SummaryWriter(args.log_path)
    env = gym.make(args.env_name)
    env.unwrapped.max_steps = args.max_steps  # define max steps
    env.unwrapped.max_episode = args.max_episode  # define max steps
    # Set seeds
    random.seed(args.seed)
    env.unwrapped.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    p_action = env.p_action_space.n

    # Initialize policy
    policy = Agent(state_dim=state_dim, action_dim=action_dim, warmup=args.warmup, writer=writer,
                 discount=args.discount, tau=args.tau, policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_freq=args.policy_freq)
    replay_buffer = ReplayBuffer()

    max_episode_avg_reward = float('-inf')
    max_episode_reward = float('-inf')
    best_reward = 0.5
    best_eval_reward = 0.5
    env.unwrapped.action_type = 3
    env.unwrapped.writer = writer
    eval_history = []
    flat_count = 0
    curr_episode = 0
    env.unwrapped.curriculum_learning = curr_episode + 50000
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
            action = policy.choose_action(state, noise_scale=args.expl_noise)
            if replay_buffer.ready(batch_size=args.batch_size):
                policy.learn(replay_buffer, args.batch_size)

            next_state, reward, done, info = env.step(action)
            total_reward += reward  # total step reward
            replay_buffer.add((state, next_state, action, reward, done))
            state = next_state

            episode_timesteps += 1
            reward_history.append(total_reward)

            print('------------------------------------------------------')
            print(f"Episode: {curr_episode} Step: {episode_timesteps} "
                  f"Reward: {reward:.5f}  --  Wallclk T: {int(time.time() - t0)} sec")
            print('------------------------------------------------------')

        max_episode_avg_reward = max(total_reward/(episode_timesteps + 1), max_episode_avg_reward)
        max_episode_reward = max(total_reward, max_episode_reward)

        if best_reward <= total_reward/(episode_timesteps + 1):
            env.unwrapped.record = True
            best_reward = total_reward/(episode_timesteps + 1)
            mean_r = evaluate(policy=policy)
            eval_history.append(mean_r)
            policy.save()
            # Save best model
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r
                if env.cache_video:
                    # Create unique filename using timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = f"{args.record_path}/debug_{timestamp}_episode{curr_episode}_action_{env.unwrapped.action_type}.mp4"
                    debug_clip = ImageSequenceClip(env.cache_video, fps=60)
                    debug_clip.write_videofile(video_path, codec="libx264")

        env.unwrapped.record = False

        episode_reward += total_reward
        writer.add_scalar('Reward/average reward', np.mean(reward_history[-100:]), curr_episode)
        writer.add_scalar('Reward/episode reward', episode_reward, curr_episode)
        writer.add_scalar('Reward/one episode reward', total_reward, curr_episode)
        writer.add_scalar('Reward/max episode average reward', max_episode_avg_reward, curr_episode)
        writer.add_scalar('Reward/max episode reward', max_episode_reward, curr_episode)
        writer.add_scalar(f"Action_{env.unwrapped.action_type}/max episode average reward", max_episode_avg_reward, curr_episode)

        print('------------------------------------------------------')
        print(f"Episode: {curr_episode} Total reward: {total_reward:.5f}  --  Wallclk T: {int(time.time() - t0)} sec")
        print('------------------------------------------------------')
        curr_episode += 1
