from __future__ import print_function

from math import log

from .utility import init_matrix, init_3d_matrix

class HiddenMarkovModel:
    """
    Notation used:
        HMM: Hidden Markov Model
        O: Observation sequence
        S: Hidden state sequence
        A: State transition probability distribution matrix
        B: Observation emission probability distribution matrix
        pi: Initial state probability distribution vector
        lambda: A HMM comprised of (A,B,pi)
    """
    def __init__(self, A, B, pi, all_obs, all_states, single_states=None, order=1):
        if(single_states == None):
            self._single_states = all_states
        else:
            self._single_states = single_states
        self._all_states = all_states
        self._all_obs = all_obs
        self._A = A
        self._B = B
        self._pi = pi
        self._highest_order = order

    def evaluate(self, sequence):
        """
        Evaluation Problem: Calculate P(O|lambda).
            Calculates the probability of emitting the given observation
            sequence based on the HMM. Uses the forward algorithm.
        Args:
            sequence (list<char>): observation sequence O
        Returns:
            float: probability of sequence being emitted
        """
        if(len(sequence) == 0):
            return 0

        alpha = self._forward(sequence)
        fwd_probability = sum(map(
            lambda s: alpha[s][len(sequence) - 1],
            range(len(self._all_states)))
        )
        return fwd_probability

    def decode(self, sequence):
        """
        Decoding Problem: Given O and lambda, find S such that S 'best'
            describes O using lambda. Uses the Viterbi Algorithm.
        Args:
            sequence (list<char>): observation sequence O
        Returns:
            list<string>: hidden state sequence S
        """
        if(len(sequence) == 0):
            return []
        return self._viterbi(sequence)

    def learn(self, sequences, delta=0.0001, k_smoothing=0.0, iterations=-1):
        """
        Learning Problem: Reestimate the model parameters (A,B,pi) iteratively
            using the Baum-Welch Algorithm (EM). Maximize P(O|lambda).
        It should be known that pi is currently not fully updated for HMMs
            of order greater than one.
        Args:
            sequences (list<O>): list of observations O = (O1,O2,...On) used
                to train the initial (A,B,pi) parameters.
            delta (float): log value of iterative improvement such that when
                evaluation probabilities improve by less than delta the
                learning process is complete.
            k_smoothing (float): Smoothing parameter for add-k smoothing to
                avoid zero probability. Value should be between [0.0, 1.0].
            iterations (int): number of iterations to perform. Will return
                if convergence is found before all iterations
                have been performed.
        Returns:
            (int): number of iterations to achieve convergence.
        """
        num_sequences = len(sequences)

        cur_iterations = 0
        if(num_sequences == 0):
            return cur_iterations

        prior_score = sum(map(
            lambda O: log(self.evaluate(O)),
            sequences
        )) / num_sequences

        while True:
            map(lambda O: self._train(O, k_smoothing), sequences)
            cur_iterations += 1

            new_score = sum(map(
                lambda O: log(self.evaluate(O)),
                sequences
            )) / num_sequences

            if(abs(prior_score - new_score) < delta):
                break
            if(iterations > -1 and cur_iterations >= iterations):
                break
            prior_score = new_score

        return cur_iterations

    def get_parameters(self):
        """ Dictionary of all model parameters. """
        return {
            "A": self._A,
            "B": self._B,
            "pi": self._pi,
            "all_obs": self._all_obs,
            "all_states": self._all_states,
            "single_states": self._single_states
        }

    def display_parameters(self):
        """ Display the lambda parameters (A,B,pi) on the console. """
        names = [
            "Starting probabilities (pi):",
            "Transition probabilities (A):",
            "Emission probabilities (B):"
        ]
        for i, parameter in enumerate([self._pi, self._A, self._B]):
            print(names[i])
            for element in parameter:
                print(element)

    # ----------------- #
    #      Private      #
    # ----------------- #

    def _forward(self, sequence):
        rows = len(self._all_states)
        columns = len(sequence)
        alpha = init_matrix(rows, columns, "float")

        # initialization step
        for s_index, state in enumerate(self._single_states):
            o_index = self._all_obs.index(sequence[0])
            alpha[s_index][0] = (
                self._pi[0][state]
                * self._B[s_index][o_index]
            )

        # iterative step
        for t_index in range(columns - 1):
            obs = sequence[t_index + 1]
            for s_index, state in enumerate(self._all_states):
                single_state_index = self._single_states.index(
                    self._get_state_by_order(state, 1)
                )
                for s_prime in range(len(self._all_states)):
                    if(t_index + 1 < self._highest_order):
                        state_by_order = self._get_state_by_order(
                            self._all_states[s_index],
                            t_index + 2
                        )
                        a_prob = self._pi[t_index + 1][state_by_order]
                    else:
                        a_prob = self._A[s_prime][s_index]

                    alpha[s_index][t_index + 1] += (
                        alpha[s_prime][t_index]
                        * a_prob
                        * self._B[single_state_index][self._all_obs.index(obs)]
                    )

        return alpha

    def _backward(self, sequence):
        rows = len(self._all_states)
        columns = len(sequence)
        beta = init_matrix(rows, columns, "float")

        # initialization step
        for s_index, state in enumerate(self._all_states):
            beta[s_index][-1] = 1

        # iterative step
        for t_index in reversed(range(columns-1)):
            obs = sequence[t_index + 1]
            for s_index in range(len(self._all_states)):
                for s_prime, state in enumerate(self._all_states):
                    single_state_index = self._single_states.index(
                        self._get_state_by_order(state, 1)
                    )
                    beta[s_index][t_index] += (
                        beta[s_prime][t_index + 1]
                        * self._A[s_index][s_prime]
                        * self._B[single_state_index][self._all_obs.index(obs)]
                    )

        return beta

    def _viterbi(self, sequence):
        """
        Notation used:
            delta: matrix holding the highest probability state path
                at observation time t.
            psi: backpointer matrix maintaining which state maximized delta.
        Args:
            sequence (list<char>): observation sequence O
        Returns:
            list<string>: hidden state sequence S
        """
        delta, psi = self._viterbi_forward(sequence)
        return self._viterbi_backward(delta, psi, sequence)

    def _viterbi_forward(self, sequence):
        """ build probability quantities delta and backpointers psi """
        rows = len(self._all_states)
        columns = len(sequence)

        delta = init_matrix(rows, columns, "int")
        psi = init_matrix(rows, columns, 'int,int')

        # initialization step
        obs_index = self._all_obs.index(sequence[0])
        for s_index, state in enumerate(self._all_states):
            single_state = self._get_state_by_order(state, 1)
            single_state_index = self._single_states.index(single_state)
            delta[s_index][0] = (
                self._pi[0][single_state]
                * self._B[single_state_index][obs_index]
            )

        # iterative step
        for o_index in range(1, columns):
            o_master_index = self._all_obs.index(sequence[o_index])
            for s_index, state in enumerate(self._all_states):
                max_prob = 0
                row_back = 0
                col_back = 0

                single_state_index = self._single_states.index(self._get_state_by_order(state, 1))
                emission_multiplier = self._B[single_state_index][o_master_index]

                # a multiplier of 0.0 nullfies the following computation
                if emission_multiplier == 0.0:
                    continue

                for prev_s_index in range(rows):
                    transition_multiplier = 0
                    if(o_index < self._highest_order):
                        state_by_order = self._get_state_by_order(
                            self._all_states[s_index],
                            o_index + 1
                        )
                        transition_multiplier = self._pi[o_index][state_by_order]
                    else:
                        transition_multiplier = self._A[prev_s_index][s_index]

                    cur_prob = (
                        delta[prev_s_index][o_index - 1]
                        * transition_multiplier
                        * emission_multiplier
                    )
                    if cur_prob > max_prob:
                        max_prob = cur_prob
                        row_back = prev_s_index
                        col_back = o_index - 1

                delta[s_index][o_index] = max_prob
                psi[s_index][o_index] = (row_back, col_back)

        return delta, psi

    def _viterbi_backward(self, delta, psi, sequence):
        """ Decode by following the backpointers of psi """
        rev_output = []
        j_max = len(sequence)
        max_final = 0
        i_final = 0

        # find highest probability start state
        for i in range(len(self._all_states)):
            current_final = delta[i][j_max - 1]
            if current_final > max_final:
                max_final = current_final
                i_final = i

        rev_output.append(self._get_state_by_order(self._all_states[i_final], 1))
        i_cur = psi[i_final][j_max - 1][0]
        j_cur = psi[i_final][j_max - 1][1]

        for j in range(j_max - 2, -1, -1):
            rev_output.append(self._get_state_by_order(self._all_states[i_cur], 1))
            i_cur_old = i_cur
            i_cur = psi[i_cur][j_cur][0]
            j_cur = psi[i_cur_old][j_cur][1]

        return rev_output[::-1]

    def _train(self, sequence, k_smoothing=0.0):
        """
        Use the Baum-Welch Algorithm which utilizes Expectation-Maximization
        and the Forward-Backward algorithm to find the maximum likelihood
        estimate for parameters (A,B,pi).
        Notation used:
            gamma: Probability of being in state i at time t
                given O and (A,B,pi).
                Row: state. Column: observation
            xi: Joint probability of being in state i at time t and
                state (i + 1) at time (t + 1) given O and (A,B,pi).
                xi[state i][state j][time t]
        Args:
            sequence (list<char>): Observation sequence O
            k_smoothing (float): Smoothing parameter for add-k smoothing to
                avoid zero probability. Value should be between [0.0, 1.0].
        """
        rows = len(self._all_states)
        columns = len(sequence)

        alpha = self._forward(sequence)
        beta = self._backward(sequence)

        # build gamma
        gamma = init_matrix(rows, columns, "float")
        for s_index in range(rows):
            for o_index in range(columns):
                prob = alpha[s_index][o_index] * beta[s_index][o_index]
                prob /= sum(map(
                    lambda j: alpha[j][o_index] * beta[j][o_index],
                    range(rows)
                ))
                gamma[s_index][o_index] = prob

        # buid xi
        xi = init_3d_matrix(rows, rows, columns - 1, "float")
        for o_index in range(columns - 1):
            obs = sequence[o_index]
            obs_next = sequence[o_index + 1]

            denominator = 0.0
            for s_from in range(rows):
                for s_to, state_to in enumerate(self._all_states):
                    single_state_index = self._single_states.index(
                        self._get_state_by_order(state_to, 1)
                    )
                    prob = (
                        alpha[s_from][o_index]
                        * beta[s_to][o_index + 1]
                        * self._A[s_from][s_to]
                        * self._B[single_state_index][self._all_obs.index(obs_next)]
                    )
                    xi[s_from][s_to][o_index] = prob
                    denominator += prob

            if denominator == 0:
                continue

            for s_from in range(rows):
                for s_to in range(rows):
                    xi[s_from][s_to][o_index] /= denominator

        # update all parameters (A,B,pi).
        for s_index, state in enumerate(self._all_states):
            # update pi
            self._pi[self._highest_order - 1][state] = (
                (gamma[s_index][0] + k_smoothing)
                / (1 + rows * k_smoothing)
            )

            # update A
            gamma_sum = sum(map(
                lambda o_index: gamma[s_index][o_index],
                range(columns - 1)
            ))
            if(gamma_sum == 0):
                for s_prime in range(rows):
                    self._A[s_index][s_prime] = 0
            else:
                for s_prime in range(rows):
                    xi_sum = sum(map(
                        lambda o_index: xi[s_index][s_prime][o_index],
                        range(columns - 1)
                    ))
                    self._A[s_index][s_prime] = (
                        (xi_sum + k_smoothing)
                        / (gamma_sum + (rows * k_smoothing))
                    )

            # update B
            gamma_sum += gamma[s_index][columns - 1]
            single_state_index = self._single_states.index(
                self._get_state_by_order(state, 1)
            )
            if(gamma_sum == 0):
                for o_index in range(columns):
                    self._B[single_state_index][o_index] = 0
            else:
                gamma_b_sum = list(map(
                    lambda x: 0,
                    range(len(self._all_obs))
                ))

                for o_index in range(columns):
                    full_obs_index = self._all_obs.index(sequence[o_index])
                    gamma_b_sum[full_obs_index] += gamma[s_index][o_index]

                for o_index in range(len(self._all_obs)):
                    self._B[single_state_index][o_index] = (
                        (gamma_b_sum[o_index] + k_smoothing)
                        / (gamma_sum + (columns * k_smoothing))
                    )

    def _get_state_by_order(self, state, order):
        """
        Gets single state for any order HMM.
        Examples (let order == 1):
            'a1-b0-a0' => 'a0'
            'a1-b0' => 'b0'
            'a1' => 'a1'
        Args:
            state: '-' delimited composite state
            order: desired order state to return
        Returns:
            string: modified state
        """
        if(self._highest_order == 1):
            return state
        split_state = state.split('-')
        l = len(split_state)
        if(order > l):
            raise ValueError("Specified order is higher than given state.")

        return '-'.join(split_state[l - order:l])
