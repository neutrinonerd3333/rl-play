import typing

import collections
import random
import numpy

import keras.models
import keras.layers
import keras.backend
import keras.utils.np_utils
import tensorflow

# LOTS of inspiration from https://github.com/matthiasplappert/keras-rl/

class LearnerMemory:
    def __init__(self):
        self.history = []

    def append(self, item):
        self.history.append(item)

    def sample(self, sample_size):
        return random.sample(self.history, min(sample_size, len(self.history)))


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
        self.history = LearnerMemory()
        self.batch_size = batch_size

    def best_action(self, state: str) -> int:
        return numpy.argmax(self.table[state])

    def update(self, old_state, new_state, action, reward, terminal, gamma, **kwargs):
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
        learning_rate = kwargs["learning_rate"]
        assert 0 <= learning_rate <= 1
        assert 0 <= gamma < 1

        # experience replay
        if self.batch_size is not None:
            self.history.append((old_state, new_state, action, reward, terminal))

            experience = self.history.sample(self.batch_size)
            olds, news, acts, rewards, terminalness = [[tup[i] for tup in experience] for i in range(5)]

            old_q = numpy.array([self.table[old][act] for (old, _, act, _, _) in experience])
            expected_futures = gamma * numpy.array([numpy.max(self.table[new]) for new in news])
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


class DeepQNetwork(BaseQApproximator):
    def __init__(self, model, batch_size=None, delta_clip=numpy.inf):
        self.history = LearnerMemory()
        self.batch_size = batch_size
        self.delta_clip = delta_clip

        # input model
        self.model = model
        inputs = model.inputs
        outputs = model.outputs
        y_pred_tensor = outputs[0]

        # target (actor) model
        config = {
            'class_name': model.__class__.__name__,
            'config': model.get_config(),
        }
        self.target_model = keras.models.model_from_config(config)

        self.model.compile(optimizer='sgd', loss='mse')
        self.target_model.compile(optimizer='sgd', loss='mse')


        # assert len(outputs) == 1
        # print(keras.backend.int_shape(y_pred_tensor))
        # assert len(keras.backend.int_shape(y_pred_tensor)) == 1
        self.action_n = keras.backend.int_shape(y_pred_tensor)[1]
        y_true_tensor = keras.layers.Input(name='y_true', shape=(self.action_n,))
        action_tensor = keras.layers.Input(name='action_mask', shape=(self.action_n,))

        def huber_loss(error, clip):
            squared_loss = keras.backend.square(error) / 2
            if numpy.isinf(clip):
                return squared_loss
            
            condition = keras.backend.abs(x) < clip
            linear_loss = clip * (K.abs(x) - clip / 2)

            return tensorflow.select(condition, squared_loss, linear_loss)

        def huber_loss_function(args):
            y_true, y_pred, mask = args
            errors = y_true - y_pred
            losses = huber_loss(errors, self.delta_clip)
            return keras.backend.sum(losses * mask, axis=-1)

        def identity(y_true, y_pred):
            return y_pred

        loss_tensor = keras.layers.Lambda(huber_loss_function, output_shape=(1,), name='loss')([y_true_tensor, y_pred_tensor, action_tensor])
        self.trainable_model = keras.models.Model(input=(inputs + [y_true_tensor, action_tensor]), output=loss_tensor)
        self.trainable_model.compile(loss=identity, optimizer='sgd')

    def best_action(self, state):
        np_state = numpy.array(state).reshape(1, -1)
        q_vals = self.target_model.predict(np_state)
        # print("Q-vals in state {}: {}".format(state, q_vals))
        # print("best action: {}".format(numpy.argmax(q_vals)))
        return numpy.argmax(q_vals)

    def update(self, old_state, new_state, action, reward, terminal, gamma, **kwargs):
        assert 0 <= gamma < 1

        # add to history
        self.history.append((old_state, new_state, action, reward, terminal))
        
        experience = self.history.sample(self.batch_size)
        cur_batch_size = len(experience)
        olds, news, acts, rewards, terminalness = [numpy.array([tup[i] for tup in experience]) for i in range(5)]

        discounted_futures = gamma * numpy.max(self.target_model.predict_on_batch(news), axis=1)
        q_vals = rewards + numpy.logical_not(terminalness) * discounted_futures
        q_val_array = numpy.zeros((cur_batch_size, self.action_n))
        for ind in range(cur_batch_size):
            q_val_array[ind][acts[ind]] = q_vals[ind]

        # update our networks
        acts_one_hot = keras.utils.np_utils.to_categorical(acts, self.action_n)
        assert acts_one_hot.shape == (cur_batch_size, self.action_n)

        dummy_targets = numpy.zeros((cur_batch_size, 1))

        self.trainable_model.train_on_batch([olds, q_val_array, acts_one_hot], dummy_targets)
        self.target_model.set_weights(self.model.get_weights())


