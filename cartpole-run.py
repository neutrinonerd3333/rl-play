import time
import argparse

import numpy
import matplotlib.pyplot as plt
import pandas

import keras.models
from keras.regularizers import l2

import gym
import gym.spaces

from qlearner import QLearner
from approximators import DeepQNetwork, TabularQApproximator

# bin_params = [(3, 2.4), (4, 1.5), (8, 0.27), (6, 1.5)]
bin_params = [(3, 2.4), (3, 1.5), (12, 0.27), (12, 1.5)]
cart_pole_bins = [pandas.cut([-bound, bound],
                             bins=bin_n,
                             retbins=True)[1][1:-1]
                  for (bin_n, bound) in bin_params]


def build_state(observation):
    obs_bins_pairs = zip(observation, cart_pole_bins)
    return tuple(numpy.digitize(x=obs, bins=bins)
                   for (obs, bins) in obs_bins_pairs)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO include allowed envs
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="OpenAI Gym environment name.")
    parser.add_argument("--monitor", action="store_true",
                        help="Whether to use the Gym monitor. "
                             "Requires FFMpeg with libx264.")
    parser.add_argument("--plot", action="store_true",
                        help="Plot our progress")
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help="Verbosity. 1 for episode details, "
                             "2 for Q-details.")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train for.")

    # discount factor
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Initial discount factor "
                             "for utility calculations.")
    parser.add_argument("--gamma-final", type=float, default=None,
                        help="\u03b3, post-annealing.")

    # learning rate
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Initial learning rate. "
                             "Only applicable for tabular Q-learning.")
    parser.add_argument("--alpha-final", type=float, default=0.1,
                        help="Learning rate, post-annealing.")

    # exploration parameter
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial value of \u03b5 as in \u03b5-greedy.")
    parser.add_argument("--epsilon-final", type=float, default=0.01,
                        help="\u03b5, post-annealing.")

    # annealing
    parser.add_argument("--anneal", type=int, default=100,
                        help="Number of episodes over which to anneal.")

    # experience replay
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Specify a minibatch size "
                             "to enable experience replay. "
                             "Required for deep Q-networks.")

    # all things deep
    parser.add_argument("--deep", action="store_true",
                        help="Use a deep Q-network.")
    parser.add_argument("--delta-clip", type=float, default=10,
                        help="Gradient clipping threshold "
                             "for DQN Huber loss.")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[32],
                        help="Numbers of nodes in hidden layers.")

    args = parser.parse_args()

    # env specific
    env_name = args.env
    env = gym.make(env_name)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    action_n = env.action_space.n
    observation_n = env.observation_space.shape[0]

    victory_thresh = 0
    past_n = 0
    if env_name == "CartPole-v0":
        victory_thresh = 195
        past_n = 100
    elif env_name == "CartPole-v1":
        victory_thresh = 475
        past_n = 100
    elif env_name == "LunarLander-v2":
        victory_thresh = 200
        past_n = 100
    elif env_name == "MountainCar-v0":
        victory_thres = -110.0
        past_n = 100
    else:
        raise ValueError("Environment {} not supported".format(env_name))

    # monitoring
    if args.monitor:
        time_str = time.strftime("%Y%m%d_%H%M%S")
        env.monitor.start('/tmp/{}-{}'.format(env_name, time_str), force=True)

    # learning setup
    if args.deep:
        # build the model
        first_hidden_layer = keras.layers.Dense(args.hidden_layers[0],
                                                input_shape=(observation_n,),
                                                activation='relu',
                                                W_regularizer=l2(0.01))
        other_hidden_layers = [
            keras.layers.Dense(n_nodes,
                               activation='relu',
                               W_regularizer=l2(0.01))
            for n_nodes in args.hidden_layers[1:]]
        output_layer = keras.layers.Dense(action_n)

        layers = [first_hidden_layer] + other_hidden_layers + [output_layer]
        print("layers: {}".format(layers))
        model = keras.models.Sequential(layers)

        # and the approximator
        if not isinstance(args.batch_size, int):
            raise ValueError("Must specify --batch-size with --deep!")
        approximator = DeepQNetwork(model,
                                    batch_size=args.batch_size,
                                    delta_clip=args.delta_clip)
    else:
        approximator = TabularQApproximator(action_n,
                                            batch_size=args.batch_size)

    final_gamma = args.gamma if args.gamma_final is None else args.gamma_final
    learner = QLearner(action_n,
                       approximator,
                       args.gamma,
                       gamma_final=final_gamma,
                       learning_rate=args.alpha,
                       learning_rate_final=args.alpha_final,
                       epsilon=args.epsilon,
                       epsilon_final=args.epsilon_final,
                       annealing_time=args.anneal)

    n_episodes = args.episodes

    # stats
    episodes = []
    rewards = []
    past_n_avg = []
    ewmas = []
    ewma_factor = 0.1

    plt.ion()
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # LEARN!
    for i_episode in range(n_episodes):
        observation = env.reset()
        episode_reward = 0
        num_timesteps = 0
        done = False

        while not done:
            env.render()
            num_timesteps += 1

            state = observation if args.deep else build_state(observation)

            # q-learning
            action = learner.select_action(state, verbose=(args.verbose >= 2))
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            new_state = observation if args.deep else build_state(observation)
            learner.update_q(state, new_state, action, reward, done)

        # exponentially weighted moving avg
        new_ewma = episode_reward if len(ewmas) == 0 \
            else ewmas[-1] * (1 - ewma_factor) + ewma_factor * episode_reward

        # stats
        episodes.append(i_episode)
        rewards.append(episode_reward)
        past_n_avg.append(numpy.mean(rewards[-past_n:]))
        ewmas.append(new_ewma)

        if args.verbose >= 1:
            report_str = "Episode {} took {} timesteps, reward {}, " \
                         "\u03b5 = {:.4}, \u03b3 = {:.4}"
            print(report_str.format(i_episode + 1,
                                    num_timesteps,
                                    episode_reward,
                                    learner.current_epsilon,
                                    learner.current_gamma))

        learner.anneal(i_episode)

        # plotting, monitoring
        if i_episode % 10 == 0:
            plt.plot(episodes, rewards,
                     color="cornflowerblue",
                     label='rewards')
            plt.plot(episodes, ewmas,
                     color="mediumorchid",
                     label='ewma, $\\alpha=0.1$')
            plt.plot(episodes, past_n_avg,
                     color="darkorchid",
                     label='avg of past {}'.format(past_n))
            # plt.legend()
            plt.draw()

        if past_n_avg[-1] > victory_thresh:
            print("Success!")
            break

    plt.show()

    if args.monitor:
        env.monitor.close()


if __name__ == '__main__':
    main()
