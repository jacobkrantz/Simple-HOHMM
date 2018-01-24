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
        for do_synthesize in [True, False]:
            for order in range(1, 5):
                hmm = builder.build(
                    highest_order=order,
                    k_smoothing=.01,
                    synthesize_states=do_synthesize
                )
                self._test_parameters(hmm.get_parameters(), order)

    def test_build_synthesize(self):
        builder = Builder()
        builder.add_batch_training_examples(self._obs, self._states)
        hmm_synth = builder.build(
            highest_order=3,
            k_smoothing=.01,
            synthesize_states=True
        )
        hmm_no_synth = builder.build(
            highest_order=3,
            k_smoothing=.01,
            synthesize_states=False
        )
        params_synth = hmm_synth.get_parameters()
        params = hmm_no_synth.get_parameters()
        # there should be more possible starting states with params_synth
        self.assertGreater(len(params_synth["pi"][2]), len(params["pi"][2]))
        # there should be more possible state transitions with params_synth
        self.assertGreater(len(params_synth["A"]), len(params["A"]))
        self.assertEqual(len(params_synth["B"]), len(params["B"]))

    def test_set_states_before_build(self):
        builder = Builder()
        builder.add_batch_training_examples(self._obs, self._states)
        builder.set_all_obs(['normal', 'cold', 'dizzy'])
        builder.set_single_states(['fever', 'healthy', 'blah'])
        hmm = builder.build(
            highest_order=2,
            k_smoothing=.01,
            synthesize_states=False
        )
        hmm2 = builder.build(
            highest_order=2,
            k_smoothing=.01,
            synthesize_states=True
        )
        self._test_parameters(hmm.get_parameters(), 2)
        self.assertEqual(hmm.get_parameters(), hmm2.get_parameters())

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

    def _test_parameters(self, params, order):
        for value in params.values():
            self.assertIsNotNone(value)
        for i in range(order):
            self.assertAlmostEqual(sum(params["pi"][i].values()), 1)
        self.assertLessEqual(
            len(params["single_states"]),
            len(params["all_states"])
        )
        if(order > 1):
            return

        for i in range(2):
            self.assertAlmostEqual(sum(params["A"][i]), 1)
        for i in range(2):
            self.assertAlmostEqual(sum(params["B"][i]), 1)
