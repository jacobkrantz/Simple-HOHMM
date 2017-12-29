import unittest

from SimpleHOHMM import HiddenMarkovModel as HMM

class TestHMM(unittest.TestCase):

    def setUp(self):
        all_observations = ['normal', 'cold', 'dizzy']
        all_states = ['healthy', 'fever']
        start_probs = [{"healthy": 0.6, "fever": 0.4}]

        trans_probs = [
            [0.7, 0.3],
            [0.4, 0.6]
        ]

        emission_probs = [
            [0.5, 0.4, 0.1],
            [0.1, 0.3, 0.6]
        ]

        self._hmm = HMM(
            trans_probs,
            emission_probs,
            start_probs,
            all_observations,
            all_states
        )
        self._sequence = ['normal', 'cold', 'dizzy', 'dizzy','cold','normal']

    def tearDown(self):
        self._hmm = None
        self._sequence = None

    def test_hmm_evaluate(self):
        eval = self._hmm.evaluate(self._sequence)
        self.assertGreater(eval, 0)
        self.assertLess(eval, 1)

    def test_hmm_decode(self):
        decoded = self._hmm.decode(self._sequence)
        self.assertEqual(len(decoded), len(self._sequence))
        for state in decoded:
            self.assertFalse(state in self._sequence)

    def test_hmm_learn(self):
        sequences = [
            ['normal', 'cold', 'dizzy','normal','normal'],
            ['normal', 'cold', 'normal','dizzy','normal'],
            ['dizzy', 'dizzy', 'dizzy','cold','normal'],
            ['dizzy', 'dizzy', 'normal','normal','normal'],
            ['cold', 'cold', 'dizzy','normal','normal'],
            ['normal', 'dizzy', 'dizzy','normal','cold'],
        ]
        num_iterations = self._hmm.learn(sequences, k_smoothing=0.005)
        self.assertGreater(num_iterations, 0)
        self.test_hmm_evaluate()
        self.test_hmm_decode()
