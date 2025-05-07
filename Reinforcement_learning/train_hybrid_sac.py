import argparse
import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from networks import Policy, SoftQNetwork, ReplayBuffer
from utils import to_gym_action, to_torch_action, gym_to_buffer
from wrappers.goal_wrappers import ScaledStateWrapper, GoalFlattenedActionWrapper, ScaledParameterisedActionWrapper, \
    GoalObservationWrapper
import gym
from gym.wrappers import Monitor
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    parser.add_argument('--exp-name', type=str, default="sac_experiment", help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Goal-v0", help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4, help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=4000000, help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False,
                        help='run in production mode with wandb')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False,
                        help='whether to capture videos')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL", help="wandb project name")
    parser.add_argument('--wandb-entity', type=str, default=None, help="wandb entity")
    parser.add_argument('--autotune', type=lambda x: bool(strtobool(x)), default=True, help='automatic entropy tuning')
    parser.add_argument('--buffer-size', type=int, default=20000, help='replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1, help="timesteps to update target network")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size for replay memory")
    parser.add_argument('--tau', type=float, default=0.1, help="target smoothing coefficient")
    parser.add_argument('--alpha', type=float, default=0.2, help="entropy regularization coefficient")
    parser.add_argument('--learning-starts', type=int, default=5000, help="timestep to start learning")
    parser.add_argument('--policy-lr', type=float, default=1e-4, help='policy network learning rate')
    parser.add_argument('--q-lr', type=float, default=1e-3, help='Q network learning rate')
    parser.add_argument('--policy-frequency', type=int, default=1, help='actor update delay')
    parser.add_argument('--weights-init', default='kaiming', choices=['xavier', 'orthogonal', 'kaiming'],
                        help='weight initialization')
    parser.add_argument('--bias-init', default='zeros', choices=['zeros', 'uniform'], help='bias initialization')
    parser.add_argument('--ent-c', type=float, default=-0.99, help='target entropy continuous')
    parser.add_argument('--ent-d', type=float, default=0.1498, help='target entropy discrete')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters',
                    "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    if args.prod_mode:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
                   name=experiment_name, monitor_gym=True, save_code=True)
        writer = SummaryWriter(f"/tmp/{experiment_name}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    env = gym.make(args.gym_id)
    env = GoalObservationWrapper(env)
    env = GoalFlattenedActionWrapper(env)
    env = ScaledParameterisedActionWrapper(env)
    env = ScaledStateWrapper(env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    if args.capture_video:
        env = Monitor(env, f'videos/{experiment_name}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    input_shape, out_c, out_d = 17, 4, 3
    rb = ReplayBuffer(args.buffer_size)
    pg = Policy(input_shape, out_c, out_d, args).to(device)
    qf1 = SoftQNetwork(input_shape, out_c, out_d, args).to(device)
    qf2 = SoftQNetwork(input_shape, out_c, out_d, args).to(device)
    qf1_target = SoftQNetwork(input_shape, out_c, out_d, args).to(device)
    qf2_target = SoftQNetwork(input_shape, out_c, out_d, args).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    policy_optimizer = optim.Adam(pg.parameters(), lr=args.policy_lr)
    loss_fn = nn.MSELoss()

    if args.autotune:
        target_entropy, target_entropy_d = args.ent_c, args.ent_d
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        log_alpha_d = torch.zeros(1, requires_grad=True, device=device)
        alpha, alpha_d = log_alpha.exp().item(), log_alpha_d.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=1e-4)
        a_d_optimizer = optim.Adam([log_alpha_d], lr=1e-4)
    else:
        alpha, alpha_d = args.alpha, args.alpha

    global_episode, num_goals, (obs, _), done = 0, 0, env.reset(), False
    episode_reward, episode_length = 0.0, 0

    for global_step in range(1, args.total_timesteps + 1):
        if global_step < args.learning_starts:
            action_ = env.action_space.sample()
            action_ = gym_to_buffer(action_)
            action = [action_[0], action_[1:]]
        else:
            action_c, action_d, _, _, _ = pg.get_action([obs], device)
            action = to_gym_action(action_c, action_d)

        (next_obs, _), reward, done, _ = env.step(action)
        rb.put((obs, gym_to_buffer(action), reward / 50.0, next_obs, done))
        episode_reward += reward
        episode_length += 1
        obs = np.array(next_obs)

        if len(rb.buffer) > args.batch_size:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions_c, next_state_actions_d, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = pg.get_action(
                    s_next_obses, device)
                qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions_c, device)
                qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions_c, device)
                min_qf_next_target = next_state_prob_d * (torch.min(qf1_next_target,
                                                                    qf2_next_target) - alpha * next_state_prob_d * next_state_log_pi_c - alpha_d * next_state_log_pi_d)
                next_q_value = torch.Tensor(s_rewards).to(device) + (
                            1 - torch.Tensor(s_dones).to(device)) * args.gamma * min_qf_next_target.sum(1).view(-1)

            s_actions_c, s_actions_d = to_torch_action(s_actions, device)
            qf1_a_values = qf1.forward(s_obs, s_actions_c, device).gather(1, s_actions_d.long().view(-1,
                                                                                                     1)).squeeze().view(
                -1)
            qf2_a_values = qf2.forward(s_obs, s_actions_c, device).gather(1, s_actions_d.long().view(-1,
                                                                                                     1)).squeeze().view(
                -1)
            qf1_loss = loss_fn(qf1_a_values, next_q_value)
            qf2_loss = loss_fn(qf2_a_values, next_q_value)
            qf_loss = (qf1_loss + qf2_loss) / 2

            values_optimizer.zero_grad()
            qf_loss.backward()
            values_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    actions_c, actions_d, log_pi_c, log_pi_d, prob_d = pg.get_action(s_obs, device)
                    qf1_pi = qf1.forward(s_obs, actions_c, device)
                    qf2_pi = qf2.forward(s_obs, actions_c, device)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    policy_loss_d = (prob_d * (alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
                    policy_loss_c = (prob_d * (alpha * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
                    policy_loss = policy_loss_d + policy_loss_c

                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, _, lpi_c, lpi_d, p_d = pg.get_action(s_obs, device)
                        alpha_loss = (-log_alpha * p_d * (p_d * lpi_c + target_entropy)).sum(1).mean()
                        alpha_d_loss = (-log_alpha_d * p_d * (lpi_d + target_entropy_d)).sum(1).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                        a_d_optimizer.zero_grad()
                        alpha_d_loss.backward()
                        a_d_optimizer.step()
                        alpha_d = log_alpha_d.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/alpha_d", alpha_d, global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                    writer.add_scalar("losses/alpha_d_loss", alpha_d_loss.item(), global_step)

        if done:
            global_episode += 1
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)
            writer.add_scalar("charts/episode_length", episode_length, global_step)
            if global_episode % 10 == 0:
                print(f"Episode: {global_episode} Step: {global_step}, Ep. Reward: {episode_reward}")
            if int(reward) == 50:
                num_goals += 1
            if global_episode % 100 == 0:
                writer.add_scalar("charts/p_goal", num_goals / 100, global_step)
                num_goals = 0
            (obs, _), done = env.reset(), False
            episode_reward, episode_length = 0.0, 0

    writer.close()
    env.close()
    torch.save(pg.state_dict(), 'goal.pth')