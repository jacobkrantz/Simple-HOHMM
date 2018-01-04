import unittest

from SimpleHOHMM import HiddenMarkovModelBuilder as Builder

class TestHMMBuilder(unittest.TestCase):

    def setUp(self):
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
        self._obs = None
        self._states = None

    def test_build(self):
        builder = Builder()
        builder.add_batch_training_examples(self._obs, self._states)
        for order in range(1, 5):
            hmm = builder.build(highest_order=order, k_smoothing=.01)
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

    def test_build_uniform(self):
        builder = Builder()
        builder.set_all_obs(['normal', 'cold', 'dizzy'])
        builder.set_single_states(['healthy', 'fever'])
        uniform_hmm = builder.build_unsupervised(distribution="uniform")
        uniform_hmm_2 = builder.build_unsupervised(distribution="uniform")
        self.assertEqual(
            uniform_hmm.get_parameters(),
            uniform_hmm_2.get_parameters()
        )

        params = uniform_hmm.get_parameters()
        self.assertEqual(len(set(params["pi"][0].values())), 1)
        for row in params["A"]:
            self.assertEqual(len(set(row)), 1)
            self.assertAlmostEqual(sum(row), 1)
        for row in params["B"]:
            self.assertEqual(len(set(row)), 1)
            self.assertAlmostEqual(sum(row), 1)

    def test_build_random(self):
        builder = Builder()
        builder.set_all_obs(['normal', 'cold', 'dizzy'])
        builder.set_single_states(['healthy', 'fever'])
        random_hmm = builder.build_unsupervised(distribution="random")
        random_hmm_2 = builder.build_unsupervised(distribution="random")
        self.assertNotEqual( # ignore small chance they could be the same
            random_hmm.get_parameters(),
            random_hmm_2.get_parameters()
        )

        params = random_hmm.get_parameters()
        self.assertAlmostEqual(sum(params["pi"][0].values()), 1)
        self.assertGreater(len(params["A"]), 1)
        for row in params["A"]:
            self.assertGreater(len(row), 1)
            self.assertAlmostEqual(sum(row), 1)

        self.assertGreater(len(params["B"]), 1)
        for row in params["B"]:
            self.assertGreater(len(row), 1)
            self.assertAlmostEqual(sum(row), 1)
