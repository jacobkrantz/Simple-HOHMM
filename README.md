# Simple-HOHMM  

[![Build Status](https://travis-ci.org/jacobkrantz/Simple-HOHMM.svg?branch=master)](https://travis-ci.org/jacobkrantz/Simple-HOHMM)
[![Coverage Status](https://coveralls.io/repos/github/jacobkrantz/Simple-HOHMM/badge.svg?branch=master)](https://coveralls.io/github/jacobkrantz/Simple-HOHMM?branch=master)  

Simple-HOHMM is an end-to-end sequence classifier using Hidden Markov Models. Let the builder construct a model for you based on chosen model attributes. Now you can solve the classic problems of HMMs: evaluating, decoding, and learning. Play with different orders of history to maximize the accuracy of your model!

## General

#### Solving Fundamental Problems
* **Evaluation**  
	Given an observation sequence and an HMM, determine the probability that the HMM would emit that exact observation sequence. Done with the *Forward Algorithm*.
* **Decoding**  
	Given an observation sequence and an HMM, determine the most likely hidden state sequence that would emit the observation sequence. Done with the *Viterbi Algorithm*.
* **Learning**  
	Given a set of observation sequences and an HMM, reestimate the model parameters so as to maximize the probabilities resulting from the Evaluation problem. Done with the *Baum Welch EM Algorithm*.

#### Features
* Learning is done in any manner desired: **supervised**, **semi-supervised**, or **unsupervised**. Supervised is done with training examples of explicit counts. Semi-supervised is generated with some examples followed by a learning algorithm. Unsupervised is done by creating a model of either uniformly or randomly distributed parameters followed by a learning algorithm.
* Discrete (Multinomial) emissions only.
* Ergotic state transitions are assumed by the model, but setting certain probabilities to zero effectively emulates unreachable states.
* Smoothing of model parameters is done with additive k-smoothing to avoid cases of zero probability, especially useful for higher order modeling.
* `HiddenMarkovModel` can be trained using `HiddenMarkovModelBuiler` or by passing in explicit HMM parameter values.

## Getting Started

#### Requirements
This project is currently written in pure python code with zero dependencies. Code has been tested and runs with both `Python 2.7 ` and `Python 3.5`. Running with [pypy](https://pypy.org/) offers drastic speed improvements, consider this when working with large models.

#### Installing Simple-HOHMM
No official distribution currently exists for this project as the source code is still in flux. Know that a distribution is coming. To use the code now, just clone the repository:  
`$ git clone https://github.com/jacobkrantz/Simple-HOHMM.git`  

#### Documentation  
Documentation consisting of API reference and basic tutorials will be hosted using [ReadTheDocs](https://readthedocs.org/). This not been developed yet. For now we detail a basic example to get you up and running. Soon you will be able to take a more in-depth look at advanced modeling options.  
#### Example problem
(adapted from Wikipedia)  
Suppose villagers are either healthy or have a fever. Fevers are diagnosed by the doctor asking patients how they feel (normal, dizzy, or cold). Assuming their health can be modeled by a discrete Markov chain, the observations are `(normal, dizzy, cold)` and the hidden states are `(healthy, fever)`. The doctor has seen patients in the past, and kept that data. The observations are in one list and the states are in another such that `states[i]` corresponds to `observations[i]`:  
```python
observations = [
  ['normal', 'cold', 'dizzy', 'dizzy','normal','normal'],
  ['cold', 'cold', 'dizzy', 'normal','normal','normal'],
  ['dizzy', 'dizzy', 'cold', 'normal', 'dizzy', 'normal'],
  ['normal', 'normal', 'cold', 'dizzy', 'dizzy', 'dizzy']
]
states = [
	['healthy', 'healthy', 'fever', 'fever', 'healthy', 'healthy'],
	['healthy', 'fever', 'fever', 'healthy', 'healthy', 'fever'],
	['fever', 'fever', 'fever', 'healthy', 'healthy', 'healthy'],
	['healthy', 'healthy', 'healthy', 'fever', 'fever', 'fever']
]
```
We can now build a first order Hidden Markov Model based on the observations and states above:
```python
from SimpleHOHMM import HiddenMarkovModelBuilder as Builder
builder = Builder()
builder.add_batch_training_examples(observations, states)
hmm = builder.build()
```
Now suppose a patient has been seeing the doctor for three days and felt `(normal, cold, dizzy)`. What might the doctor guess about this patient's health? This is solved with Viterbi decoding:  
```python
obs =  ['normal', 'cold', 'dizzy']
states = hmm.decode(obs)
print(states) # prints: ['healthy', 'healthy', 'fever']
```
