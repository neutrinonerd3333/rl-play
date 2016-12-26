import collections
import random

import numpy
import sklearn.neural_network

class QLearner:
    def __init__(self, action_n,
                 gamma,
                 learning_rate=0.8,
                 epsilon=0.6,
                 learning_rate_decay=1,
                 learning_rate_decay_delay=150,
                 epsilon_decay=0.99,
                 epsilon_decay_delay=150):
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

        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_delay = epsilon_decay_delay
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_delay = learning_rate_decay_delay

        self.current_epsilon = epsilon
        self.current_learning_rate = learning_rate

# inspired by https://gym.openai.com/algorithms/alg_0eUHoAktRVWWM7ZoDBWQ9w
class TabularQLearner(QLearner):
    def __init__(self, action_n,
                 gamma,
                 learning_rate=0.8,
                 epsilon=0.6,
                 learning_rate_decay=1,
                 learning_rate_decay_delay=150,
                 epsilon_decay=0.99,
                 epsilon_decay_delay=150):
        """
        @param gamma: utility discount factor
        @param learning_rate: learning rate of algorithm
        @param epsilon: as in ``epsilon-greedy"
        """
        # preconditions
        assert 0 < gamma < 1
        assert 0 < learning_rate <= 1
        assert 0 < epsilon <= 1

        super().__init__(action_n, gamma, learning_rate, epsilon, learning_rate_decay, learning_rate_decay_delay, epsilon_decay, epsilon_decay_delay)
        self.qtable = collections.defaultdict(lambda: numpy.random.normal(0, 0.1, self.action_n))

    def select_action(self, state):
        """
        @param state: our current state
        @return an epsilon-greedy choice
        """
        greedy = (random.random() > self.current_epsilon)
        return numpy.argmax(self.qtable[state]) \
            if greedy else random.randrange(self.action_n)

    def update_q(self, old_state, new_state, action, reward):
        """
        Update our table of Q values.

        @param old_state: the old state
        @param new_state: the new state
        @param action: the action taken
        @param reward: the reward we got for the action
        """
        old_q_val = self.qtable[old_state][action]
        self.qtable[old_state][action] = \
            (1 - self.current_learning_rate) * old_q_val + \
            self.current_learning_rate * (reward + self.gamma * numpy.max(self.qtable[new_state]))

    def decay(self, i_episode):
        self.current_epsilon = self.epsilon if i_episode < self.epsilon_decay_delay \
            else max(0.05, self.current_epsilon * self.epsilon_decay)
        self.current_learning_rate = self.learning_rate if i_episode < self.learning_rate_decay_delay \
            else max(0.1, self.current_learning_rate * self.learning_rate_decay)

class DeepQLearnerDiscrete(QLearner):
    def __init__(self, action_n,
                 gamma,
                 learning_rate,
                 epsilon=0.6,
                 learning_rate_decay=1,
                 learning_rate_decay_delay=150,
                 epsilon_decay=0.99,
                 epsilon_decay_delay=150,
                 hidden_layer_sizes=(100,)):
        super().__init__(action_n, gamma, learning_rate, epsilon, learning_rate_decay, learning_rate_decay_delay, epsilon_decay, epsilon_decay_delay)
        self.q_net = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, learning_rate='adaptive')

    def select_action(self, state):
        greedy = (random.random() > self.current_epsilon)
        return numpy.argmax(self.q_net.predict(state.reshape(1, -1))) \
            if greedy else random.randrange(self.action_n)

    def update_q(self, old_state, new_state, action, reward):
        try:
            correct_q_val = reward + self.gamma * numpy.max(self.q_net.predict(new_state.reshape(1, -1)))
            current_q_array = self.q_net.predict(old_state.reshape(1, -1))
            current_q_array[0][action] = correct_q_val
        except sklearn.exceptions.NotFittedError:
            current_q_array = numpy.random.normal(0, 1, (1, self.action_n))
            current_q_array[0][action] = reward

        self.q_net.partial_fit(old_state.reshape(1, -1), current_q_array.reshape(1, -1))

    def decay(self, i_episode):
        self.current_epsilon = self.epsilon if i_episode < self.epsilon_decay_delay \
            else max(0.05, self.current_epsilon * self.epsilon_decay)
