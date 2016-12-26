import time
import argparse

import numpy
import matplotlib.pyplot
import pandas

import gym
import gym.spaces

from qlearner import TabularQLearner, DeepQLearnerDiscrete

cart_pole_bins = [pandas.cut([-bound, bound], bins=bin_n, retbins=True)[1][1:-1] for (bin_n, bound) in [(3, 2.4), (4, 1.5), (8, 0.27), (6, 1.5)]]
def build_state(observation):
    obs_bins_pairs = zip(observation, cart_pole_bins)
    discretized = [numpy.digitize(x=obs, bins=bins) for (obs, bins) in obs_bins_pairs]
    return ";".join(map(str, discretized))

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--env", type=str, default="CartPole-v0")  # 'MountainCar-v0'
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)

    # learning rate
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--alpha_decay", type=float, default=0.99)
    parser.add_argument("--alpha_decay_delay", type=int, default=150)

    # exploration parameter
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--epsilon_decay", type=float, default=0.99)
    parser.add_argument("--epsilon_decay_delay", type=int, default=150)

    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[100])

    parser.add_argument("--episodes", type=int, default=1000)

    args = parser.parse_args()

    # env specific
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    action_n = env.action_space.n

    # monitoring
    if args.monitor:
        time_str = time.strftime("%Y%m%d_%H%M%S")
        env.monitor.start('/tmp/{}-{}'.format(env_name, time_str), force=True)

    # learning setup
    learner = TabularQLearner(action_n, args.gamma,
                              learning_rate=args.alpha,
                              epsilon=args.epsilon,
                              learning_rate_decay=args.alpha_decay,
                              learning_rate_decay_delay=args.alpha_decay_delay,
                              epsilon_decay=args.epsilon_decay,
                              epsilon_decay_delay=args.epsilon_decay_delay) \
        if not args.deep else \
        DeepQLearnerDiscrete(action_n, args.gamma,
                              learning_rate=args.alpha,
                              epsilon=args.epsilon,
                              learning_rate_decay=args.alpha_decay,
                              learning_rate_decay_delay=args.alpha_decay_delay,
                              epsilon_decay=args.epsilon_decay,
                              epsilon_decay_delay=args.epsilon_decay_delay,
                              hidden_layer_sizes=args.hidden_layers)

    n_epsiodes = args.episodes
    episodes = numpy.array([])
    timesteps = numpy.array([])
    past_hundred_avg = numpy.array([])

    MAX_T = 100000

    ewmas = []
    ewma_factor = 0.1

    matplotlib.pyplot.ion()

    # LEARN!
    for i_episode in range(n_epsiodes):
        observation = env.reset()
        for t in range(MAX_T):
            env.render()
            # print(observation)

            state = observation if args.deep else build_state(observation)

            # select actions
            action = learner.select_action(state)
            observation, reward, done, info = env.step(action)
            new_state = observation if args.deep else build_state(observation)
            learner.update_q(state, new_state, action, reward, done)

            if done or t == MAX_T - 1:
                num_timesteps = t + 1
                episodes = numpy.append(episodes, i_episode)
                timesteps = numpy.append(timesteps, num_timesteps)
                past_hundred_avg = numpy.append(past_hundred_avg, numpy.mean(timesteps[-100:]))

                # exponentially weighted moving avg
                if len(ewmas) == 0:
                    ewmas.append(num_timesteps)
                else:
                    ewmas.append(ewmas[-1] * (1 - ewma_factor) + ewma_factor * num_timesteps)

                print("Episode {} finished after {} timesteps, {} = {:.4}, {} = {:.4}"\
                    .format(i_episode + 1, num_timesteps, chr(949),
                        learner.current_epsilon, chr(945), learner.current_learning_rate))
                learner.decay(i_episode)
                break

        # plotting, monitoring
        if i_episode % 10 == 0:
            matplotlib.pyplot.plot(episodes, timesteps, color="cornflowerblue")
            matplotlib.pyplot.plot(episodes, ewmas, color="mediumorchid")
            matplotlib.pyplot.plot(episodes, past_hundred_avg, color="darkorchid")
            matplotlib.pyplot.draw()

        if past_hundred_avg[-1] > 195:
            print("Success!")
            break

    matplotlib.pyplot.show()

    if args.monitor:
        env.monitor.close()


if __name__ == '__main__':
    main()
