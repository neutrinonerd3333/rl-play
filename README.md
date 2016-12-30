# rl-play

(I apologize in advance for silly typos and grammar mistakes
in this README.
I'll proofread this someday :/)

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


## Just let me run the code already

If you're just starting out with this code,
you probably want to see it work really well.
In that case, run
```
python cartpole-run.py --plot --epsilon 1 --epsilon-final .01 --anneal 100 --episodes 2000 --batch-size 32 --hidden-layers 32 --gamma 0.9 --gamma-final 0.95 --deep -v
```
or better yet:
```
python cartpole-run.py --plot --epsilon 1 --epsilon-final .01 --anneal 200 --episodes 2000 --batch-size 64 --gamma 0.8 --gamma-final 0.98 --deep -v --env "CartPole-v1"
```
Read on to find out what all these flags signify.

## Example usage

The command
```
python cartpole-run.py -v
```
runs the tabular Q-learning algorithm
in its most basic incarnation with default parameters
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

To increase the episode limit beyond the default 1000,
just use the `--episodes` flag:
```
python cartpole-run.py -v --episodes 1337
```

We can also set training hyperparameters with flags:
```
python cartpole-run.py --alpha 0.6 --epsilon 0.1 --anneal 100
```
(see below for details).

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
python cartpole-run.py --alpha 0.8 --alpha-final 0.1 --anneal 100
```
and say that we let α linearly *anneal* over 100 episodes from 0.8 to 0.1.

We might also want to anneal ε,
since we might want an agent to have an exploration-heavy start
before slowly shifting toward exploitation.
To do so, use the `--epsilon` and `--epsilon-final` flags:
```
python cartpole-run.py --alpha 0.8 --alpha-final 0.1 --epsilon 1 --epsilon-final 0.01 --anneal 100 --plot -v
```

In fact, α and ε are annealed by default.
Flags like `--alpha` merely let you specify
*initial* values of hyperparameters.
To get rid of annealing for α,
just set `--alpha-final` to be the same as `--alpha`.
We can do the same with ε.

You can also anneal γ,
but the code does not by default.
You shouldn't really have to do so in the tabular case.
Indeed, γ has an intuitive interpretation
as a proxy for the expected lifetime of the agent per episode,
with the characteristic lifetime given by (1 - γ)^(-1).
So for an agent we expect to live for 100 turns,
we should approximately set γ = 0.99.

Moral: we should think of γ less as a hyperparameter
and more as a parameter given to use by the environment we're in.

Note that the annealing time specified by `--anneal`
is *shared* by all parameters that we anneal:
it seems that annealing all parameters together
works reasonably well in practice.
(I might change this behavior in the future.)

Still, our agent still learns quite slowly.
One way to fix this issue is to use *experience replay*,
where we store up our experience
(roughly speaking: the states we transitioned from and to,
the action we chose,
and the reward we earned)
and use it to update our Q tables later on.
To enable experience replay,
specify `--batch-size` with the amount of experience
we'd like to replay on each timestep.
A good about is 32:
```
python cartpole-run.py --alpha 0.8 --alpha-final 0.1 --epsilon 1 --epsilon-final 0.01 --anneal 100 --batch-size 32 --plot -v
```

### Deep Q-learning

It's straightforward to do deep Q-learning:
just add the `--deep` flag.
γ and ε still work as before.
The deep Q-network doesn't care about α;
currently, we use one of Keras' stochastic gradient descent optimizers with weight decay.

With deep Q-learning, it's important to use experience replay.
Neural net training implicitly rests on the assumption
that the data we feed in is i.i.d.,
which definitely isn't the case if we naively ran the Q-learning algorithm.
Thus, the code requires the `--batch-size` with `--deep`.

To specify the architecture of the neural net,
use the `--hidden-layers` flag.
If we want two hidden layers with 50 and 10 nodes, respectively,
we would write
```
python cartpole-run.py --deep --batch-size 32 --plot -v --hidden-layers 50 10
```

Once we switch over to using a neural net,
it's very easy to run into divergence issues;
to see divergence very quickly, use a large value of γ:
```
python cartpole-run.py --plot --anneal 100 --batch-size 32 --gamma 0.999 --deep -vv
```
(The `-vv` flag puts us in extra-verbose mode,
letting us check on the values produced by the network.)

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
python cartpole-run.py -v --plot --gamma 0.8 --gamma-final 0.95 --anneal 100 --deep --batch-size 32
```

The deep Q-network is trained with [Huber loss](https://en.wikipedia.org/wiki/Huber_loss),
a variant of squared error
in which gradients are clipped at some threshold,
which can be specified as
```
python cartpole-run.py -v --deep --delta-clip 10 --batch-size 32
```
Gradient clipping keeps our network robust to outliers,
which is important in the sorts of noisy environments
our agent is likely to encounter.


## wishlist

* Double Q learning to combat overoptimism and divergence issues caused by positive feedback loops; with double Q-learning we could potentially use more representative values of γ.
* prioritized experience replay
