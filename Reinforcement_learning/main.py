import numpy as np
import torch
import gym
import env
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time
from utils.RelayBuffer import ReplayBuffer
# from models.TD3 import TD3
from models.OurTD3 import TD3


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=5):
    avg_reward = 0.
    ids = float('inf') # sample id/training trick
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(state))
            actions = [action, ids]
            next_state, reward, done, info = env.step(actions)
            state = next_state
            ids = info['id']
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--start_timesteps", default=5000,
                        type=int)  # How many time steps purely random policy is run for (10#-30%)
    parser.add_argument("--eval_freq", default=5e4, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--log_path", default='./reward_log', type=str)  # reward log path
    parser.add_argument("--max_episode", default=1e5, type=float)  # Max episode to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic (recommend 128)
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()
    print("---------------------------------------")
    print("*WARN*: Have you activated checkpoint saving mode yet?")
    print("---------------------------------------")

    file_name = "%s_%s" % (args.policy_name, args.env_name)
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if os.path.isdir("./results"):
        shutil.rmtree("./results")
    os.makedirs("./results")

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if os.path.isdir("./pytorch_models"):
        shutil.rmtree("./pytorch_models")
    os.makedirs("./pytorch_models")

    writer = SummaryWriter(args.log_path)
    env = gym.make(args.env_name)

    MIN_REPLAY_SIZE = 800 #number of sample experiences (recommend 800)
    # MIN_REPLAY_SIZE = arg.batch_size * 5 #Time to collect sample experiments
    state_dim = env.observation_space.shape[0] #512 + 6 + 6 - instruction, joints, coordinates
    action_dim = env.action_space.shape[0] # 6 joints
    max_action = float(env.action_space.high[0])
    p_action = env.p_action_space.n # 5 possible actions
    suction = env.suction_space.n # 2 suction states

    # Initialize policy
    policy = TD3(state_dim, action_dim, p_action, suction, max_action)
    replay_buffer = ReplayBuffer()

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)]

    # ids = float('inf')  # sample id/training trick
    # for step in range(MIN_REPLAY_SIZE):
    #     state = env.reset()
    #     if step < args.start_timesteps:
    #         action = np.array(env.action_space.sample())
    #         p_action = np.array([env.p_action_space.sample()])
    #         suction = np.array([env.suction_space.sample()])
    #     else:
    #         action, p_action, suction = policy.select_action(np.array(state))
    #         if args.expl_noise != 0:
    #             action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
    #                 env.action_space.low, env.action_space.high)
    #     actions = [action, p_action, suction]
    #     actions = [actions, ids]
    #     # Perform action
    #     next_state, reward, done, info = env.step(actions)
    #     # Store data in replay buffer
    #     replay_buffer.add((info['id'], state, next_state, *actions[0], reward, done))

    curr_episode = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    reward_history = []
    success = []
    t0 = time.time()

    '''TRAINING PROCESS'''
    while curr_episode < args.max_episode:

        total_reward = 0
        episode_timesteps = 0
        update = False
        state = env.reset()
        ids = float('inf')  # Sample ids/ training trick
        done = False

        while not done:
            update = True
            if curr_episode < args.start_timesteps:
                action = np.array(env.action_space.sample())
                p_action = np.array([env.p_action_space.sample()])
                suction = np.array([env.suction_space.sample()])
            else:
                action, p_action, suction = policy.select_action(np.array(state))
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                        env.action_space.low, env.action_space.high)
            actions = [action, p_action, suction]
            actions = [actions, ids]
            next_state, reward, done, info = env.step(actions)
            done_bool = 1 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            total_reward += reward
            replay_buffer.add((info['id'], state, next_state, *actions[0], reward, done_bool))
            state = next_state
            ids = info['id']
            policy.train(replay_buffer, args.batch_size, args.discount, args.tau,
                         args.policy_noise, args.noise_clip, args.policy_freq)
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy))
                if args.save_models:
                    policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)
            episode_timesteps += 1
            reward_history.append(total_reward)
        episode_reward += total_reward
        if total_reward >= 0 and update == True:
            success.append(1)
        else:
            success.append(0)
        if curr_episode % 1000 == 0 and curr_episode != 0:
            print('Successful Rate: ', sum(success[-100:]), '%')
            writer.add_scalar('success_rate', sum(success[-100:]), curr_episode)
        writer.add_scalar('avg reward', np.mean(reward_history[-100:]), curr_episode)
        writer.add_scalar('episode reward', episode_reward, curr_episode)
        curr_episode += 1
        timesteps_since_eval += 1
        print(f"Episode: {curr_episode} Step: {episode_timesteps} "
              f"Reward: {total_reward:.6f}  --  Wallclk T: {int(time.time() - t0)} sec")

    # Final evaluation
    evaluations.append(evaluate_policy(policy))
    if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)
