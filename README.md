# rl-play

This repository contains code I wrote to play with reinforcement learning.
The code is far from production quality, but it works! (Mostly.)
Currently, the code supports tabular and deep Q-learning.

## Dependencies

This code requires:
* numpy
* matplotlib
* tensorflow
* keras
* tensorflow

All of these dependencies can be `pip install`-ed.

## Example usage

The command
```
python cartpole-run.py -v
```
runs the tabular Q-learning algorithm with default parameters
on the default CartPole-v0 environment,
printing out per-episode information.

Adding the `--plot` flag
```
python cartpole-run.py -v --plot
```
will give us a plot of reward as a function of episode
(an episode is a single play-through of whatever game we're playing).
The resulting plot also displays
the exponentially-weighted moving average
and the past-100 episode average of the reward plotted alongside.

To train on a different environment,
use the `--env` flag:
```
python cartpole-run.py -v --env CartPole-v1
```
Currently, the learning code only works well on the CartPole environments,
although MountainCar-v0 is also supported.

To train for more episodes, just use the `--episodes` flag:
```
python cartpole-run.py -v --episodes 1337
```

We can also set training hyperparameters with flags:
```
python cartpole-run.py --alpha 0.6 --epsilon 0.1 --anneal 100
```
see below for details.

Running
```
python cartpole-run.py -h
```
gives you a useful help message :)

## Details

If you're not familiar with Q-learning,
consult [the Wikipedia page](https://en.wikipedia.org/wiki/Q-learning)
or [my notes from MIT's 6.867](http://web.mit.edu/txz/www/links.html).

### Tabular Q-learning

In its most basic form, the tabular Q-learning algorithm
with ε-greedy action selection
takes the following hyperparameters,
all real numbers between 0 and 1:
* γ (a utility discount factor)
* α (a learning rate)
* ε (as in "ε-greedy")

All these can be specified:
```
python cartpole-run.py --gamma 0.99 --alpha 0.8 --epsilon 0.1
```

Choosing hyperparameters is tricky business,
as anyone who has had to wrangle with neural nets knows.
In particular, α is, as all learning rates are,
very finicky.
Much like learning rates in stochastic gradient descent,
we can ask α to *decay*.
Our code implements a linear decay.
To ask that α decay from 0.8 to 0.1 over 100 episodes,
we can write
```
python cartpole-run.py --alpha 0.8 --alpha-min 0.1 --anneal 100
```
and say that we let α linearly *anneal* over 100 episodes from 0.8 to 0.1.

We can also anneal ε in the same way.
Indeed, we might want our agent to have an exploration-heavy start
and slowly shift toward exploitation.
In fact, α and ε are annealed by default;
flags like `--alpha` merely let you specify
*initial* values of hyperparameters.
To get rid of annealing for α,
just set `--alpha-min` to be the same as `--alpha`.
We can do the same with ε.

You can also anneal γ,
but the code does not by default.
You shouldn't really have to do so in the tabular case.
Indeed, γ has an intuitive interpretation
as a proxy for the expected lifetime of the agent per episode,
with the characteristic lifetime given by (1 - γ)^(-1).
So for an agent we expect to live for 100 turns,
we should approximately set γ = 0.99.

Note that the annealing time specified by `--anneal`
is *shared* by all parameters that we anneal:
it seems that annealing all parameters together
works reasonably well in practice.

### Deep Q-learning

It's straightforward to do deep Q-learning:
just add the `--deep` flag.
γ and ε still work as before.
The deep Q-network doesn't care about α;
currently, we use one of Keras' stochastic gradient descent optimizers with weight decay.

To specify the architecture of the neural net,
use the `--hidden-layers` flag.
If we want two hidden layers with 50 and 10 nodes, respectively,
we would write
```
python cartpole-run.py -plot -v --hidden-layers 50 10
```

Once we switch over to using a neural net,
it's very easy to run into divergence issues.
If you get a warning from numpy saying
that multiplication isn't defined for the given operands,
it's likely that the deep Q-network is producing `nan` values.
(You can check the output of the Q-network
by running in extra-verbose mode with the `-vv` flag.)

We could solve these divergence issues
by cranking up the regularization on the network weights,
but the resulting algorithm doesn't learn quite well.
Instead, we use a smaller value of γ (0.8 seems to work well).

But smaller γ results in a more myopic agent
that seeks out short-term gain over potential long-term reward.
We'd still like to use a γ that tracks the expected length of an episode,
as described above.
Therefore, when we being training the network,
we start with smaller γ to prevent divergence,
then let γ anneal upwards as we go:
```
python cartpole-run.py -v --plot --gamma 0.8 --gamma-final 0.99 --anneal 100 --deep
```

The deep Q-network is trained with [Huber loss](https://en.wikipedia.org/wiki/Huber_loss),
a variant of squared error
in which gradients are clipped at some threshold,
which can be specified as
```
python cartpole-run.py -v --deep --delta-clip 0.5
```
Gradient clipping keeps our network robust to outliers,
which is important in the sorts of noisy environments
our agent is likely to encounter.

