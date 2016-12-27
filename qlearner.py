import typing
import random

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
        self.q_approximator.update(old_state, new_state, action, reward, terminal, self.gamma, learning_rate=self.current_learning_rate)

    def decay(self, i_episode):
        self.current_epsilon = max(self.epsilon_min, self.epsilon - i_episode * self.epsilon_decay_rate)
        self.current_learning_rate = max(self.learning_rate_min, self.learning_rate - i_episode * self.learning_rate_decay_rate)
