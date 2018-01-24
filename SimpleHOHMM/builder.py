from copy import deepcopy
from itertools import product
import random as ran

from .model import HiddenMarkovModel as HMM
from .utility import init_matrix, init_matrix_uniform, init_matrix_random

class HiddenMarkovModelBuilder:

    def __init__(self):
        self._obs_sequences = list()
        self._state_sequences = list()
        self._single_states = None
        self._all_obs = None

    def add_training_example(self, o, s):
        """
        Adds a single training example to the model builder.
        Args:
            o (list<char>): Observation sequence
            s (list<char>): Hidden state sequence
        """
        self._obs_sequences.add(o)
        self._state_sequences.add(s)

    def add_batch_training_examples(self, o_lst, s_lst):
        """
        Adds a batch of training examples to the model builder.
        Args:
            o_lst (list<list<char>>): Observation sequences
            s_lst (list<list<char>>): Hidden state sequences
        """
        self._obs_sequences += o_lst
        self._state_sequences += s_lst

    def set_single_states(self, single_states):
        """
        Sets the singular hidden states vocabulary for the HMM. If called
        multiple times, the vocabulary is overwritten.
        Args:
            single_states (list<string>): list of possible singular hidden
                states. These states should disregard HMM order.
        """
        self._single_states = single_states

    def set_all_obs(self, all_obs):
        """
        Sets the observation vocabulary for the HMM. If called multiple
        times, the vocabulary is overwritten.
        Args:
            all_obs (list<string>): list of possible model observations.
        """
        self._all_obs = all_obs

    def build(self, highest_order=1, k_smoothing=0.0, synthesize_states=False):
        """
        Builds a Hidden Markov Model based on the previously added
            training examples.
        Args:
            highest_order (int): History window of hidden states. Defaults to 1.
            k_smoothing (float): Parameter for add-k smoothing, a
                generalization of Laplace smoothing. Defaults to 0.0.
            synthesize_states (boolean): Generate all states from permutations
                of single states. Avoids OOV for higher order models and
                and ensures model is fully ergodic.
        Returns:
            HiddenMarkovModel: capable of evaluating, decoding, and learning.
        """
        if(highest_order < 1):
            raise ValueError("highest order must be 1 or greater.")

        # build state and observation sets
        if(self._all_obs is None):
            all_obs = self._get_unique_elements(self._obs_sequences)
        else:
            all_obs = self._all_obs

        if(self._single_states is None):
            single_states = self._get_higher_order_states(self._state_sequences, 1)
            if(synthesize_states):
                all_states = self._make_permutations(single_states, highest_order)
            else:
                all_states = self._get_higher_order_states(self._state_sequences, highest_order)
        else:
            synthesize_states = True
            single_states = self._single_states
            all_states = self._make_permutations(single_states, highest_order)

        # build probability distribution parameters
        start_probs = list()
        for i in range(highest_order):
            start_probs.append(self._calculate_start_probs(
                self._state_sequences,
                single_states,
                i+1,
                k_smoothing,
                synthesize_states
            ))
        trans_probs = self._calculate_transition_probs(all_states, highest_order, k_smoothing)
        emission_probs = self._calculate_emission_probs(single_states, all_obs, k_smoothing)

        # combine all parameters to build final model
        return HMM(
            trans_probs,
            emission_probs,
            start_probs,
            all_obs,
            all_states,
            single_states=single_states,
            order=highest_order
        )

    def build_unsupervised(self, single_states=None, all_obs=None, distribution="random", highest_order=1):
        """
        Builds a Hidden Markov Model based on a uniform probability
        distribution.
        Args:
            single_states (list<>): list of unique elements detailing all
                possible hidden states the model should account for. If default,
                uses the values set previously through 'set_single_states'.
            all_obs (list<>): list of unique elements detailing all possible
                observation elements the model should account for. If default,
                uses the values set previously through 'set_all_obs'.
            distribution (string): either 'random' for a random probability
                distribution, or 'uniform' for a uniform probability
                distribution. defaults to 'random'.
            highest_order (int): History window of hidden states. Defaults to 1.
        Returns:
            HiddenMarkovModel: capable of evaluating, decoding, and learning.
        """
        if(distribution not in ('random', 'uniform')):
            raise ValueError("parameter 'distribution must be either 'random' or 'uniform'")
        if(single_states is None):
            single_states = self._single_states
        if(all_obs is None):
            all_obs = self._all_obs

        single_states = list(set(single_states))
        all_obs = list(set(all_obs))
        all_states = self._make_permutations(single_states, highest_order)
        num_states = len(all_states)
        if(distribution == 'uniform'):
            trans_probs = init_matrix_uniform(num_states, num_states)
            emission_probs = init_matrix_uniform(num_states, len(all_obs))
            start_probs = self._init_uniform_start_probs(
                single_states,
                highest_order
            )
        else: # 'random'
            trans_probs = init_matrix_random(num_states, num_states)
            emission_probs = init_matrix_random(num_states, len(all_obs))
            start_probs = self._init_random_start_probs(
                single_states,
                highest_order
            )

        # combine all parameters to build final model
        return HMM(
            trans_probs,
            emission_probs,
            start_probs,
            all_obs,
            all_states,
            single_states=single_states,
            order=highest_order
        )

    def clear_all_sets(self):
        """
        Deletes all training examples previously in the builder.
        Deletes observation and hidden state vocabularies.
        """
        self._obs_sequences = list()
        self._state_sequences = list()
        self._single_states = None
        self._all_obs = None

    # ----------------- #
    #      Private      #
    # ----------------- #

    def _get_unique_elements(self, set_of_lists):
        unique_set = set()
        for obs_lst in set_of_lists:
            unique_set.update(set(obs_lst))
        return list(unique_set)

    def _calculate_transition_probs(self, all_states, order, k_smoothing):
        matrix_size = len(all_states)
        state_trans_dict = dict()

        # initialize matrix and normalization dict
        trans_probs = init_matrix(matrix_size, matrix_size, "int")
        for state in all_states:
            state_trans_dict[state] = 0

        # insert counts of transitions
        state_sequences = self._make_higher_order_states(
            self._state_sequences,
            order
        )

        for states in state_sequences:
            for i in range(1, len(states)):
                prev_index = all_states.index(states[i - 1])
                cur_index = all_states.index(states[i])
                trans_probs[prev_index][cur_index] += 1
                state_trans_dict[all_states[prev_index]] += 1

        # normalize such that for all rows sum(trans_probs[state][s0...sn]) == 1
        for prev_index in range(matrix_size):
            divisor = state_trans_dict[all_states[prev_index]]
            for cur_index in range(matrix_size):
                trans_probs[prev_index][cur_index] += k_smoothing
                trans_probs[prev_index][cur_index] /= float(
                    divisor + (matrix_size * k_smoothing)
                )

        return trans_probs

    def _calculate_emission_probs(self, all_states, all_obs, k_smoothing):
        rows = len(all_states)
        columns = len(all_obs)
        state_emission_dict = dict()

        # initializate matrix and normalization dict
        emission_probs = init_matrix(rows, columns, "int")
        for state in all_states:
            state_emission_dict[state] = 0 + k_smoothing

        # insert counts of emissions
        for i in range(len(self._obs_sequences)):
            obs_lst = self._obs_sequences[i]
            states_lst = self._state_sequences[i]
            for j in range(len(obs_lst)):
                obs = obs_lst[j]
                obs_index = all_obs.index(obs)

                state = states_lst[j]
                state_index = all_states.index(state)

                emission_probs[state_index][obs_index] += 1
                state_emission_dict[state] += 1

        # normalize such that for all rows sum(emission_probs[state][o0...on]) == 1
        for row in range(rows):
            divisor = float(state_emission_dict[all_states[row]])
            for column in range(columns):
                emission_probs[row][column] += k_smoothing
                emission_probs[row][column] /= float(
                    divisor + (rows * k_smoothing)
                )

        return emission_probs

    def _get_higher_order_states(self, state_sequences, order):
        if(order == 1):
            return self._get_unique_elements(state_sequences)

        all_states_set = set()

        for sequence in state_sequences:
            if(len(sequence) <= order):
                continue

            for i in range(order - 1, len(sequence)):
                state = ""
                for j in range(i-order+1, i+1):
                    state += (sequence[j] + '-')

                all_states_set.add(state[:len(state)-1])

        return list(all_states_set)

    def _calculate_start_probs(self, state_sequences, single_states, order, k_smoothing, synthesize_states):
        """
        Calculates the starting probability distribution for a given order.
        Args:
            state_sequences (list<list<char>>): Hidden state sequences
            single_states (list<string>): list of possible singular hidden
                states. These states should disregard HMM order.
            order (int): History window of hidden states.
            k_smoothing (float): Parameter for add-k smoothing, a
                generalization of Laplace smoothing.
            synthesize_states (boolean): if True, creates states
        Returns:
            dict[state:probability]
        """
        start_probs_dict = dict()

        # initialize dictionary to state:0
        if synthesize_states:
            states = self._make_permutations(single_states, order)
        else:
            states = self._get_higher_order_states(state_sequences, order)
        for state in states:
            start_probs_dict[state] = 0 + k_smoothing

        # insert counts
        start_state_emissions = 0
        for state_seq in state_sequences:
            if(len(state_seq) < order):
                continue

            state = ""
            for i in range(order):
                state += (state_seq[i] + '-')
            start_probs_dict[state[:len(state)-1]] += 1
            start_state_emissions += 1

        # normalize dictionary such that sum(start_probs_dict[s0...sn]) = 1
        for state in start_probs_dict.keys():
            start_probs_dict[state] /= float(
                start_state_emissions
                + (len(states) * k_smoothing)
            )

        return start_probs_dict

    def _init_uniform_start_probs(self, states, highest_order):
        start_probs = []
        for i in range(highest_order):
            start_probs_dict = dict()
            states_of_order = self._make_permutations(states, i + 1)
            value = float(1.0 / len(states_of_order))
            for i, state in enumerate(states_of_order):
                start_probs_dict[state] = value

            start_probs.append(start_probs_dict)

        return start_probs

    def _init_random_start_probs(self, states, highest_order):
        start_probs = []
        for i in range(highest_order):
            start_probs_dict = dict()
            states_of_order = self._make_permutations(states, i + 1)
            values = [ran.random() for i in range(len(states_of_order))]
            for i, state in enumerate(states_of_order):
                start_probs_dict[state] = values[i] / sum(values)

            start_probs.append(start_probs_dict)

        return start_probs

    def _make_higher_order_states(self, state_sequences, order):
        """
        Args:
            state_sequences (list<list<string>>): states to convert to a
                given order.
            order (int): n-gram value of history.
        Returns:
            list<list<string>> state_sequences mapped to n-grams.
        Example:
            state_sequences = [['a', 'b', 'c', 'd', 'e', 'f']]
            order = 1: [['a', 'b', 'c', 'd', 'e', 'f']]
            order = 2: [['a-b', 'b-c', 'c-d', 'd-e', 'e-f']]
            order = 3: [['a-b-c', 'b-c-d', 'c-d-e', 'd-e-f']]
        """
        if(order == 1):
            return state_sequences

        new_sequences = []
        for sequence in state_sequences:
            new_sequence = []
            for i in range(order-1, len(sequence)):
                state = ""
                for j in range(i-order+1, i+1):
                    state += (sequence[j] + '-')

                new_sequence.append(state[:len(state)-1])

            new_sequences.append(new_sequence)

        return new_sequences

    def _make_permutations(self, states, highest_order):
        """ makes a list of all permutation states from a single state. """
        if(highest_order == 1):
            return states

        states_lists = product(states, repeat = highest_order)
        new_states = []
        for states_lst in states_lists:
            state = ""
            for i in range(len(states_lst)):
                state += (states_lst[i] + '-')

            new_states.append(state[:len(state)-1])

        return new_states
