import unittest
from .test_builder import TestHMMBuilder
from .test_hmm import TestHMM

def test_suite():
    loader = unittest.TestLoader()

    test_classes_to_run = [TestHMMBuilder, TestHMM]
    suites_list = []

    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    return unittest.TestSuite(suites_list)
