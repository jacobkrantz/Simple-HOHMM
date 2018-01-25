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
            A=trans_probs,
            B=emission_probs,
            pi=start_probs,
            all_obs=all_observations,
            all_states=all_states
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

    def test_hmm_high_order(self):
        pi = [{
            'healthy': 0.2863247863247863,
            'fever': 0.7136752136752137
        }, {
            'healthy-healthy': 0.2855113636363636,
            'fever-fever': 0.5696022727272727,
            'healthy-fever': 0.0014204545454545455,
            'fever-healthy': 0.1434659090909091
        }]
        A = [[
            0.0024752475247524753, 0.9925742574257426,
            0.0024752475247524753, 0.0024752475247524753
        ],[
            0.0024752475247524753, 0.0024752475247524753,
            0.25, 0.745049504950495
        ],[
            0.5972222222222222, 0.3988095238095238,
            0.001984126984126984, 0.001984126984126984
        ],[
            0.0006648936170212767, 0.0006648936170212767,
            0.33311170212765956, 0.6655585106382979
        ]]
        B = [
            [0.0007127583749109052, 0.8560228082679971, 0.14326443335709194],
            [0.5711737424188371, 0.07170888333927934, 0.3571173742418837]
        ]
        all_states = [
            'healthy-healthy', 'healthy-fever',
            'fever-healthy', 'fever-fever'
        ]
        hmm = HMM(
            A=A,
            B=B,
            pi=pi,
            all_obs=['normal', 'cold', 'dizzy'],
            all_states=all_states,
            single_states=['healthy', 'fever'],
            order=2
        )
        self.assertEqual(len(hmm.decode(self._sequence)), len(self._sequence))

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
