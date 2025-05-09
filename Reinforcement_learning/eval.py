import numpy as np
import torch
import gym
import pybullet_envs
import argparse
import os
import time
from utils.RelayBuffer import ReplayBuffer
from models.TD3 import TD3

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=1, visualize=False):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            if visualize:
                env.render(mode="human")
                time.sleep(1. / 60.)
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
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--filename",
                        default="TD3_ImitationLearning-v1")  # Model filename prefix e.g. "TD3_HalfCheetahBulletEnv-v0_0"
    parser.add_argument("--eval_episodes", default=1, type=int)  # Evaluation episodes
    parser.add_argument("--visualize", action="store_true")  # Visualize or not

    args = parser.parse_args()

    filename = args.filename
    print("---------------------------------------")
    print("Settings: %s" % (filename))
    print("---------------------------------------")

    env = gym.make(args.env_name)
    if args.visualize:
        env.render(mode="human")

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    p_action = 5
    suction = 2

    # Initialize policy
    policy = TD3(state_dim, action_dim, p_action, suction, max_action)

    # Load model
    policy.load(filename, './pytorch_models/')

    # Start evaluation
    _ = evaluate_policy(policy, eval_episodes=args.eval_episodes, visualize=args.visualize)