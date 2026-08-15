import math
import unittest

from exposure.risk import risk_score


class TestRiskScore(unittest.TestCase):
    def test_empty_distribution_is_zero_risk(self):
        self.assertEqual(risk_score([]), 0.0)

    def test_single_candidate_is_conservative(self):
        # One number carries no distributional information. Claiming full exposure
        # from it would overstate the case.
        self.assertEqual(risk_score([1.0]), 0.0)

    def test_all_zero_weights_is_zero_risk(self):
        self.assertEqual(risk_score([0.0, 0.0, 0.0]), 0.0)

    def test_peaked_distribution_scores_higher_than_flat(self):
        peaked = risk_score([0.95, 0.02, 0.01, 0.01, 0.01])
        flat = risk_score([0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertGreater(peaked, flat)

    def test_uniform_distribution_scores_zero(self):
        # Maximum entropy, zero margin: no exploitable information at all.
        self.assertAlmostEqual(risk_score([0.25] * 4), 0.0, places=9)

    def test_monotonic_in_leader_mass(self):
        scores = [
            risk_score([leader] + [(1 - leader) / 4] * 4)
            for leader in (0.3, 0.5, 0.7, 0.9)
        ]
        self.assertEqual(scores, sorted(scores))

    def test_output_always_in_unit_interval(self):
        for probs in ([1.0, 0.0, 0.0], [0.5, 0.5], [0.9, 0.05, 0.03, 0.02]):
            with self.subTest(probs=probs):
                self.assertGreaterEqual(risk_score(probs), 0.0)
                self.assertLessEqual(risk_score(probs), 1.0)

    def test_unnormalized_input_is_normalized(self):
        # Logits-like weights that happen not to sum to 1 must score the same as the
        # normalized equivalent — the model side should not have to guarantee this.
        self.assertAlmostEqual(
            risk_score([9.5, 0.2, 0.1, 0.1, 0.1]),
            risk_score([9.5 / 10, 0.02, 0.01, 0.01, 0.01]),
            places=9,
        )

    def test_negative_weights_are_floored(self):
        self.assertAlmostEqual(
            risk_score([0.9, -0.1, 0.1]), risk_score([0.9, 0.0, 0.1]), places=9
        )

    def test_order_of_input_does_not_matter(self):
        self.assertAlmostEqual(
            risk_score([0.7, 0.2, 0.1]), risk_score([0.1, 0.7, 0.2]), places=9
        )

    def test_margin_and_entropy_disagree_usefully(self):
        # Two candidates splitting the mass: small margin, but low entropy because
        # everything else is ruled out. Should not score as zero.
        two_way = risk_score([0.5, 0.5, 0.0, 0.0, 0.0])
        self.assertGreater(two_way, 0.0)

    def test_entropy_normalized_by_actual_k(self):
        # A uniform distribution is maximum-entropy regardless of k, so both score 0.
        self.assertAlmostEqual(risk_score([0.5, 0.5]), 0.0, places=9)
        self.assertAlmostEqual(risk_score([0.1] * 10), 0.0, places=9)

    def test_no_nan_from_zero_probabilities(self):
        self.assertFalse(math.isnan(risk_score([0.8, 0.2, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
