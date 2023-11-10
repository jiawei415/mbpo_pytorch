import argparse
import os
import time
import gym
import torch
import numpy as np
import traceback

from config.mbpo_config import MBPO_CONFIG
from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler
from rpto_torch import env
from rpto_torch.utils import Logger, init_logger, set_global_seed


def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="AntTruncated-v0",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')
    parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
                        help='rollout max length')
    parser.add_argument('--num_epoch', type=int, default=1000, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1, metavar='A',
                        help='minimum pool size')
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--max_path_length', type=int, default=1000, metavar='A',
                        help='max length of path')


    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    # new
    parser.add_argument("--warm_start", type=int, default=0, choices=[0, 1])
    parser.add_argument("--random_start", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max_update_epoch", type=int, default=5)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--original", type=int, default=1, choices=[0, 1])
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--error_range", type=float, default=0.2)
    parser.add_argument("--logdir", type=str, default="./results/mbpo")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--alg_type", type=str, default="MBPO")
    args = parser.parse_known_args()[0]
    alg_configs = MBPO_CONFIG[args.env_name.split("-")[0]]
    parser.set_defaults(**alg_configs)
    args = parser.parse_args()
    args.task = args.env_name.split("-")[0]
    # args.logfile = f"{args.alg_type}_{args.env_name}_{args.env_seed}_{args.seed}"
    return args


@torch.no_grad()
def eval(args, agent, env_sampler):
    env_sampler.current_state = None
    sum_reward = 0
    done = False
    test_step = 0

    while (not done) and (test_step != args.max_path_length):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
        sum_reward += reward
        test_step += 1
    eval_dict = {"test_reward": sum_reward, "test_step": test_step}
    return eval_dict

def train(args, env_sampler, env_sampler_test, predict_env, agent, env_pool, model_pool, logger):
    # evaluation
    eval_dict = eval(args, agent, env_sampler_test)
    logger.record("epoch", 0)
    logger.record("total_step", 0)
    for key, val in eval_dict.items():
        logger.record(f"eval/{key}", val)
    logger.dump(0)

    best_reward = -np.inf
    total_step = 0
    train_policy_steps, train_dynamics_steps= 0, 0
    rollout_length = 1
    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch in range(1, args.num_epoch + 1):
        for _ in range(args.epoch_length):
            if total_step >= 0 and total_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                train_predict_model(args, env_pool, predict_env)
                train_dynamics_steps += 1

                new_rollout_length = set_rollout_length(args, epoch)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)

                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)

            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)
            total_step += 1

            if len(env_pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, total_step, env_pool, model_pool, agent)

        eval_dict = eval(args, agent, env_sampler_test)
        logger.record("epoch",epoch)
        logger.record("total_step", total_step)
        logger.record("update_policy_step", train_policy_steps)
        logger.record("update_dynamics_steps", train_dynamics_steps)
        for key, val in eval_dict.items():
            logger.record(f"eval/{key}", val)
        logger.dump(0)

        cur_reward = eval_dict['test_reward']
        if cur_reward > best_reward:
            save_path = logger.get_dir()
            predict_env.model.save_model(save_path)
            agent.save_model(actor_path=save_path, critic_path=save_path)
            best_reward = cur_reward
            logger.info(f"Save model on Epoch {epoch}, Best Reward {best_reward}")


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, random=args.random_start)
        env_pool.push(cur_state, action, reward, next_state, done)


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2,
                            max_epochs_since_update=args.max_update_epoch)


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    # if total_step % args.train_every_n_steps > 0:
    #     return 0

    # if train_step > args.max_train_repeat_per_step * total_step:
    #     return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state, model_next_state),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

    return args.num_train_repeat


def main(args, logger: Logger):

    # Initial environment
    env_config = {
        "version": args.version,
        "original": args.original,
        "seed": args.env_seed,
        "error_range": args.error_range,
        "log_path": logger.get_dir(),
    }
    env = gym.make(args.env_name, **env_config)
    env_test = gym.make(args.env_name, **env_config)

    # Set random seed
    set_global_seed(args.seed)
    env.seed(args.seed)
    env_test.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    logger.info(f"Actor:\n{str(agent.policy)}")
    logger.info(f"critic:\n{str(agent.critic)}")

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    
    env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay)
    logger.info(f"Dynamics:\n{str(env_model.ensemble_model)}")

    if args.warm_start and args.model_path is not None:
        for file in os.listdir(args.model_path):
            if "_2020_" in file:
                model_path = os.path.join(args.model_path, file)
                break  
        agent.load_model(model_path)
        env_model.load_model(model_path)
        logger.info(f"Load model from {model_path}.")

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name, args.model_type)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)
    env_sampler_test = EnvSampler(env_test, max_path_length=args.max_path_length)

    train(args, env_sampler, env_sampler_test, predict_env, agent, env_pool, model_pool, logger)


if __name__ == '__main__':
    args = readParser()
    logger = init_logger(args)
    main(args, logger)
    try:
        main(args, logger)
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")
