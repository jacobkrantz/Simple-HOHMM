import unittest

from SimpleHOHMM import HiddenMarkovModelBuilder as Builder

class TestHMMBuilder(unittest.TestCase):

    def setUp(self):
        self._builder = Builder()
        self._obs = [
            ['normal', 'cold', 'dizzy', 'dizzy','normal','normal'],
            ['dizzy', 'cold', 'dizzy', 'normal','normal','normal'],
            ['dizzy', 'cold', 'dizzy', 'normal','normal','normal'],
            ['normal', 'cold', 'dizzy', 'dizzy','cold','normal'],
            ['dizzy', 'dizzy', 'dizzy', 'dizzy', 'cold', 'cold'],
            ['cold', 'cold', 'cold', 'normal', 'dizzy', 'normal'],
            ['dizzy', 'normal', 'cold', 'cold', 'dizzy', 'dizzy']
        ]
        self._states = [
            ['healthy', 'healthy', 'fever', 'fever', 'healthy', 'healthy'],
            ['fever', 'fever', 'fever', 'healthy', 'healthy', 'fever'],
            ['fever', 'fever', 'fever', 'healthy', 'healthy', 'fever'],
            ['healthy', 'healthy', 'fever', 'fever', 'fever', 'healthy'],
            ['fever', 'fever', 'fever', 'fever', 'fever', 'fever'],
            ['fever', 'fever', 'fever', 'healthy', 'fever', 'healthy'],
            ['fever', 'healthy', 'fever', 'fever', 'fever', 'fever']
        ]

    def tearDown(self):
        self._builder = None

    def test_hmm_builder(self):
        self._builder.add_batch_training_examples(self._obs, self._states)
        for order in range(1, 5):
            hmm = self._builder.build(highest_order=order, k_smoothing=.01)
            params = hmm.get_parameters()

            for value in params.values():
                self.assertIsNotNone(value)
            for i in range(order):
                self.assertAlmostEqual(sum(params["pi"][i].values()), 1)
            self.assertLessEqual(
                len(params["single_states"]),
                len(params["all_states"])
            )
            if(order > 1):
                continue

            for i in range(2):
                self.assertAlmostEqual(sum(params["A"][i]), 1)
            for i in range(2):
                self.assertAlmostEqual(sum(params["B"][i]), 1)
