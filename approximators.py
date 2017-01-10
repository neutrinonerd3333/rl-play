import typing

import collections
import random
import numpy

import keras.models
import keras.layers
import keras.optimizers
import keras.backend
import keras.utils.np_utils
import tensorflow

from binaryheap import BinaryHeap

# LOTS of inspiration from https://github.com/matthiasplappert/keras-rl/


class LearnerMemory:
    def __init__(self, memory_size=1e+6):
        self.history = []
        self.memory_size = memory_size

    def append(self, item):
        if len(self.history) == self.memory_size:
            self.history.pop(0)
        self.history.append(item)

    def sample(self, sample_size):
        return random.sample(self.history, sample_size)

    def __len__(self):
        return len(self.history)


class HeapMemory(LearnerMemory):
    def __init__(self, memory_size=1e+6):
        self.history = BinaryHeap()
        self.memory_size = memory_size

    def append(self, item):
        if len(self.history) == self.memory_size:
            self.history.trim()
        self.history.insert(item, self.history.max_priority())

    def sample(self, sample_size: int):
        return self.history.sample(sample_size)

    def change_priority(self, ind, priority):
        self.history.change_priority(ind, priority)

    def sort(self):
        self.history.sort()

    def __len__(self):
        return len(self.history)


class MemoryAtom:
    """
    A wrapper for a tuple.
    """
    def __init__(self, tuple):
        self._tuple = tuple

    def tuple(self):
        return self._tuple


class BaseQApproximator:
    def best_action(self, state: str, verbose: bool) -> int:
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
    def __init__(self, action_n: int, batch_size=None) -> None:
        self.action_n = action_n
        self.table = collections.defaultdict(
            lambda: numpy.random.normal(0, 0.1, self.action_n))
        self.history = LearnerMemory()
        self.batch_size = batch_size

    def best_action(self, state: str, verbose: bool = False) -> int:
        q_table = self.table[state]
        if verbose:
            print(q_table)
        return numpy.argmax(q_table)

    def update(self, old_state: str,
               new_state: str,
               action: int,
               reward: float,
               terminal: bool,
               gamma: float,
               **kwargs):
        """
        Update our table of Q values with the Bellman equation.

        @param old_state: the old state
        @param new_state: the new state
        @param action: the action taken
        @param reward: the reward we got for the action
        @param gamma: discount factor for utility computations.
                      Must be in [0, 1)
        @param learning_rate: learning rate parameter in tabular Q-learning
                              update step. Must be in [0, 1]
        """
        # precondition
        learning_rate = kwargs["learning_rate"]
        assert 0 <= learning_rate <= 1
        assert 0 <= gamma < 1

        # experience replay
        if self.batch_size is not None:
            self.history.append(
                MemoryAtom((old_state, new_state, action, reward, terminal)))

            experience = self.history.sample(self.batch_size)
            olds, news, acts, rewards, terminalness = list(zip(map(lambda x: x.tuple, experience)))

            old_q = numpy.array([self.table[old][act]
                                 for (old, _, act, _, _) in experience])
            expected_futures = gamma * numpy.array([numpy.max(self.table[new])
                                                    for new in news])
            new_q = rewards + \
                numpy.logical_not(terminalness) * expected_futures

            updated = (1 - learning_rate) * old_q + learning_rate * new_q

            for (old, act, new_val) in zip(olds, acts, updated):
                self.table[old][act] = new_val
        else:
            old_q = self.table[old_state][action]
            new_q = reward
            if not terminal:
                new_q += gamma * numpy.max(self.table[new_state])

            self.table[old_state][action] = \
                (1 - learning_rate) * old_q + learning_rate * new_q


def huber_loss(error, clip):
    squared_loss = keras.backend.square(error) / 2
    if numpy.isinf(clip):
        return squared_loss

    condition = keras.backend.abs(error) < clip
    linear_loss = clip * (keras.backend.abs(error) - clip / 2)

    return tensorflow.select(condition, squared_loss, linear_loss)


def identity(y_true, y_pred):
    return y_pred


class DeepQNetwork(BaseQApproximator):
    def __init__(self, model: keras.models.Model,
                 batch_size: int = 32,
                 update_freq: int = 50,
                 delta_clip=numpy.inf,
                 memory_size=1e+6,
                 prioritize=False) -> None:
        self.history = HeapMemory(memory_size=memory_size) \
            if prioritize else LearnerMemory(memory_size=memory_size)
        self.batch_size = batch_size
        self.delta_clip = delta_clip
        self.prioritize = prioritize
        self._update_count = 0
        self._update_freq = update_freq

        # input model
        self.model = model
        inputs = model.inputs
        outputs = model.outputs
        y_pred_tensor = outputs[0]

        # clone the given model to get the target (actor) model
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
        y_true_tensor = keras.layers.Input(name='y_true',
                                           shape=(self.action_n,))
        action_tensor = keras.layers.Input(name='action_mask',
                                           shape=(self.action_n,))
        loss_inputs = [y_true_tensor, y_pred_tensor, action_tensor]

        def masked_huber_loss(args):
            y_true, y_pred, mask = args
            errors = y_true - y_pred
            losses = huber_loss(errors, self.delta_clip)
            return keras.backend.sum(losses * mask, axis=-1)

        loss_tensor = keras.layers.Lambda(masked_huber_loss,
                                          output_shape=(1,),
                                          name='loss')(loss_inputs)
        self.trainable_model = keras.models.Model(
            input=(inputs + [y_true_tensor, action_tensor]),
            output=loss_tensor)
        sgd_optimizer = keras.optimizers.SGD(lr=0.008, decay=1e-6)
        self.trainable_model.compile(loss=identity, optimizer=sgd_optimizer)

    def best_action(self, state, verbose=False):
        np_state = numpy.array(state).reshape(1, -1)
        q_vals = self.target_model.predict(np_state)
        # check divergence
        if numpy.any(numpy.isnan(q_vals)):
            raise RuntimeError("\033[1;31m Q-network diverged! Try smaller \u03b3? \033[0;30m")
        if verbose:
            print("Q-vals in state {}: {}".format(state, q_vals))
            print("best action: {}".format(numpy.argmax(q_vals)))
        return numpy.argmax(q_vals)

    def update(self, old_state,
               new_state,
               action,
               reward,
               terminal,
               gamma,
               **kwargs):
        assert 0 <= gamma < 1

        # add to history, increment counter
        self.history.append(MemoryAtom((old_state, new_state, action, reward, terminal)))
        self._update_count += 1

        # sample from history
        cur_batch_size = min(self.batch_size, len(self.history))
        experience = self.history.sample(cur_batch_size)
        olds, news, acts, rewards, terminalness = \
            [numpy.array([atom.tuple()[i] for atom in experience]) for i in range(5)]

        # compute target values
        discounted_futures = \
            gamma * numpy.max(self.target_model.predict_on_batch(news),
                              axis=1)
        q_vals = \
            rewards + numpy.logical_not(terminalness) * discounted_futures
        q_val_array = numpy.zeros((cur_batch_size, self.action_n))
        for ind in range(cur_batch_size):
            q_val_array[ind][acts[ind]] = q_vals[ind]

        if self.prioritize:
            current_q_vals = self.target_model.predict_on_batch(olds)
            current_q_vals_action_selected = numpy.diagonal(numpy.take(current_q_vals, acts, axis=1))
            td_errors_abs = numpy.abs(q_vals - current_q_vals_action_selected)
            for atom, td_err_abs in zip(experience, td_errors_abs):
                self.history.change_priority(atom, td_err_abs)
            if self._update_count % 1000 == 0:
                self.history.sort()

        # update our networks
        acts_one_hot = keras.utils.np_utils.to_categorical(acts,
                                                           self.action_n)
        assert acts_one_hot.shape == (cur_batch_size, self.action_n)

        dummy_targets = numpy.zeros((cur_batch_size, 1))

        self.trainable_model.train_on_batch(
            [olds, q_val_array, acts_one_hot], dummy_targets)

        if self._update_count % self._update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
