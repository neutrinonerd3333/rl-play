import typing

import collections
import random
import numpy

import sklearn.neural_network
import keras.models


class BaseQApproximator:
    def best_action(self, state: str) -> int:
        raise NotImplementedError()

    def update(self, old_state: str,
               new_state: str,
               action: int,
               reward: float,
               terminal: bool,
               gamma: float,
               learning_rate: float):
        raise NotImplementedError()


class TabularQApproximator(BaseQApproximator):
    def __init__(self, action_n, batch_size=None):
        self.action_n = action_n
        self.table = collections.defaultdict(lambda: numpy.random.normal(0, 0.1, self.action_n))
        self.history = []
        self.batch_size = batch_size

    def best_action(self, state: str) -> int:
        return numpy.argmax(self.table[state])

    def update(self, old_state, new_state, action, reward, terminal, gamma, learning_rate):
        """
        Update our table of Q values with the Bellman equation.

        @param old_state: the old state
        @param new_state: the new state
        @param action: the action taken
        @param reward: the reward we got for the action
        @param gamma: discount factor for utility computations. Must be in [0, 1)
        @param learning_rate: learning rate parameter in tabular Q-learning
                              update step. Must be in [0, 1]
        """
        # precondition
        assert 0 <= learning_rate <= 1
        assert 0 <= gamma < 1

        # experience replay
        if self.batch_size is not None:
            self.history.append((old_state, new_state, action, reward, terminal))

            experience = random.sample(self.history, min(len(self.history), self.batch_size))
            olds, acts, rewards, terminalness = [[tup[i] for tup in experience] for i in (0, 2, 3, 4)]

            old_q = numpy.array([self.table[old][act] for (old, _, act, _, _) in experience])

            expected_futures = gamma * numpy.array([numpy.max(self.table[tup[1]]) for tup in experience])
            new_q = rewards + numpy.logical_not(terminalness) * expected_futures

            updated = (1 - learning_rate) * old_q + learning_rate * new_q

            for (old, act, new_val) in zip(olds, acts, updated):
                self.table[old][act] = new_val
        else:
            old_q = self.table[old_state][action]
            new_q = reward
            if not terminal:
                new_q += gamma * numpy.max(self.table[new_state])
            
            self.table[old_state][action] = (1 - learning_rate) * old_q + learning_rate * new_q


class DeepQApproximator(BaseQApproximator):
    def __init__(self, model):
        # self.mlp = sklearn.neural_network.MLPRegressor(*args, **kwargs)
        self.history = []

    def best_action(self, state):
        pass

    def update(self, old_state, new_state, action, reward, gamma, minibatch_size):
        assert 0 <= gamma < 1

        # add to history
        self.history.append((old_state, action, reward, new_state))

        # minibatch gradient descent update
        minibatch_experience = random.sample(self.history, minibatch_size)


class QLearner:
    """
    A learner implementing all components of the Q-learning algorithm:
    epsilon-greedy action selection, 
    """

    def __init__(self, action_n: int,
                 q_approximator: BaseQApproximator,
                 gamma: float = 0.99,
                 learning_rate: float = 0.8,
                 learning_rate_min: float = 0.1,
                 epsilon: float = 0.6,
                 epsilon_min: float = 0.01,
                 annealing_time: int = 100) -> None:
        """
        @param gamma: utility discount factor
        @param learning_rate: learning rate of algorithm
        @param epsilon: as in ``epsilon-greedy"
        """
        # preconditions
        assert 0 < gamma < 1
        assert 0 < learning_rate <= 1
        assert 0 < epsilon <= 1

        # initialize
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = action_n

        self.epsilon_min = epsilon_min
        self.learning_rate_min = learning_rate_min
        # self.annealing_time = annealing_time

        self.epsilon_decay_rate = (epsilon - epsilon_min) / annealing_time
        self.learning_rate_decay_rate = (learning_rate - learning_rate_min) / annealing_time

        # self.epsilon_decay = epsilon_decay
        # self.epsilon_decay_delay = epsilon_decay_delay
        # self.learning_rate_decay = learning_rate_decay
        # self.learning_rate_decay_delay = learning_rate_decay_delay

        self.current_epsilon = epsilon
        self.current_learning_rate = learning_rate

        self.q_approximator = q_approximator

    def select_action(self, state: str) -> int:
        """
        @param state: our current state
        @return an epsilon-greedy choice
        """
        greedy = (random.random() > self.current_epsilon)
        return self.q_approximator.best_action(state) if greedy else random.randrange(self.action_n)

    def update_q(self, old_state, new_state, action, reward, terminal):
        """
        Update our table of Q values.

        @param old_state: the old state
        @param new_state: the new state
        @param action: the action taken
        @param reward: the reward we got for the action
        @param terminal: whether new_state is a terminal state
        """
        self.q_approximator.update(old_state, new_state, action, reward, terminal, self.gamma, self.current_learning_rate)

    def decay(self, i_episode):
        self.current_epsilon = max(self.epsilon_min, self.epsilon - i_episode * self.epsilon_decay_rate)
        self.current_learning_rate = max(self.learning_rate_min, self.learning_rate - i_episode * self.learning_rate_decay_rate)
