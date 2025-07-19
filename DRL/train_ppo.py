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
import random
from moviepy.editor import ImageSequenceClip
from models.ppo import Agent
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="PPO")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--eval_freq", default=4e4, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_path", default='./reward_log', type=str)  # reward log path
    parser.add_argument("--max_episode", default=1e5, type=float)  # Max episode to run environment for
    parser.add_argument("--exit_step", default=1000, type=float)  # Max episode to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--minibatch_size", default=20, type=int)  # Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic (recommend 128)
    parser.add_argument("--gae_lambda", default=0.95, type=float)  # record episode
    parser.add_argument("--policy_clip", default=0.2, type=float)  # Discount factor
    parser.add_argument("--vf_coef", default=2.0, type=float)  # max steps per eposide
    parser.add_argument("--ent_coef", default=0.001, type=float)  # Target network update rate
    parser.add_argument("--clip_coef", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--gamma", default=0.99, type=float)  # Range to clip target policy noise
    parser.add_argument("--updates_epochs", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()
    print("---------------------------------------")
    print("*WARN*: Have you activated checkpoint saving mode yet?")
    print("---------------------------------------")

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    # Initialize saving file
    for dir_path in ["./results", args.log_path, "./pytorch_models", "./record"]:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    if os.path.isfile("./reward.txt"):
        os.remove("./reward.txt")
    with open('./reward.txt', 'w') as f:
        pass

    writer = SummaryWriter(args.log_path)
    env = gym.make(args.env_name)
    # Set seeds
    random.seed(args.seed)
    env.seed(args.seed)
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
    policy = Agent(n_actions=action_dim, input_dims=state_dim, ppo_cfg=args, writer=writer)
    curr_episode = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    reward_history = []
    success = []
    t0 = time.time()

    while curr_episode < args.max_episode:
        total_reward = 0
        n_steps = 0
        state = env.reset()
        ids = float('inf')
        done = False

        for _ in range(100):
            action, prob, val = policy.choose_action(state)
            actions = [action, ids]
            next_state, reward, done, info = env.step(actions)
            info['timeouts'] = 1 if n_steps + 1 == env._max_episode_steps else 0
            total_reward += reward
            policy.remember(state['normal'], state['privileged'], action, prob, val, reward, done, info['timeouts'])

            state = next_state

            print('------------------------------------------------------')
            print(f"Episode: {curr_episode} Step: {n_steps} "
                  f"Reward: {reward:.5f}  --  Wallclk T: {int(time.time() - t0)} sec")
            print('------------------------------------------------------')
            n_steps += 1
            reward_history.append(total_reward)

        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            if args.save_models:
                policy.save(file_name, directory="./pytorch_models")
        policy.learn(iteration=curr_episode, done=done, true_done=info['timeouts'])

        if hasattr(env, 'cache_video') and env.cache_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"./record/debug_{timestamp}.mp4"
            debug_clip = ImageSequenceClip(env.cache_video, fps=60)
            debug_clip.write_videofile(video_path, codec="libx264")

        episode_reward += total_reward
        writer.add_scalar('Reward/avg reward', np.mean(reward_history[-100:]), curr_episode)
        writer.add_scalar('Reward/episode reward', episode_reward, curr_episode)
        # saving reward during training
        with open("./reward.txt", 'a') as file:
            file.write(f"{episode_reward}\n")
        curr_episode += 1
        timesteps_since_eval += 1
        print('------------------------------------------------------')
        print(f"Episode: {curr_episode} Reward: {total_reward:.5f}  --  Wallclk T: {int(time.time() - t0)} sec")
        print('------------------------------------------------------')
    # Final evaluation
    if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    writer.close()