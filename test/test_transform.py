from unittest import TestCase

from numpy.testing import assert_almost_equal

from src.transform import diff_score, sig, pairwise_probability, build_diags


class TestTransform(TestCase):
    def test_diff_score(self):
        scores = [[1., 1., 0.], [0., 1., 0]]

        expected_diffs = [[[0., 0., 1.],
                           [0., 0., 1.],
                           [-1., -1., 0.]],
                          [[0., -1., 0.],
                           [1., 0., 1.],
                           [0., -1, 0.]]]

        self.assertListEqual(diff_score(scores).tolist(), expected_diffs)

    def test_pairwise_probability(self):
        scores = [[1., 1., 0.], [0., 1., 0]]

        expected_scores = sig([[[0., 0., 1.],
                                [0., 0., 1.],
                                [-1., -1., 0.]],
                               [[0., -1., 0.],
                                [1., 0., 1.],
                                [0., -1, 0.]]])

        assert_almost_equal(pairwise_probability(scores).tolist(), expected_scores)

    def test_build_diags(self):
        pairwise_prob = [[0.5, 0.5, 0.4, 0.3],
                                [0.5, 0.5, 0.2, 0.3],
                                [0.6, 0.8, 0.5, 0.6],
                                [0.7, 0.7, 0.4, 0.5]]

        expected_A = [[0.5, -0.5, 0.0, 0.0],
                      [0.6, 0.0, -0.4, 0.0],
                      [0.7, 0.0, 0.0, -0.3],
                      [0.0, 0.8, -0.2, 0.0],
                      [0.0, 0.7, 0.0, -0.3],
                      [0.0, 0.0, 0.4, -0.6]]


        A = build_diags(pairwise_probabilities=pairwise_prob)

        self.assertListEqual(A.tolist(), expected_A)
