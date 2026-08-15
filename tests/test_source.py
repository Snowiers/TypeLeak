import unittest

from exposure.event import assemble_event
from exposure.source import TOP_K, FakeSource


class TestFakeSource(unittest.TestCase):
    def test_sample_count(self):
        self.assertEqual(len(FakeSource(seed=1).sample(12)), 12)

    def test_distributions_have_top_k_candidates(self):
        for prediction in FakeSource(seed=1).sample(20):
            self.assertEqual(len(prediction.key_topk), TOP_K)

    def test_distributions_sum_to_one(self):
        for prediction in FakeSource(seed=2).sample(20):
            self.assertAlmostEqual(sum(p for _, p in prediction.key_topk), 1.0, places=9)

    def test_distributions_are_sorted_descending(self):
        for prediction in FakeSource(seed=3).sample(20):
            probs = [p for _, p in prediction.key_topk]
            self.assertEqual(probs, sorted(probs, reverse=True))

    def test_candidate_keys_are_distinct(self):
        for prediction in FakeSource(seed=4).sample(20):
            keys = [k for k, _ in prediction.key_topk]
            self.assertEqual(len(keys), len(set(keys)))

    def test_confidence_matches_leader(self):
        for prediction in FakeSource(seed=5).sample(20):
            self.assertAlmostEqual(prediction.confidence, prediction.key_topk[0][1])

    def test_seeded_output_is_reproducible(self):
        a = [p.key_topk for p in FakeSource(seed=7).sample(10)]
        b = [p.key_topk for p in FakeSource(seed=7).sample(10)]
        self.assertEqual(a, b)

    def test_speech_present_can_be_forced(self):
        for speech in (True, False):
            with self.subTest(speech=speech):
                predictions = FakeSource(seed=8).sample(5, speech_present=speech)
                self.assertTrue(all(p.speech_present is speech for p in predictions))

    def test_produces_a_spread_of_severities(self):
        # The panels are only demoable if the fake stream distinguishes exposure
        # levels. A generator that emits one severity is useless for the alert panel.
        severities = {
            assemble_event(p).alert_severity for p in FakeSource(seed=11).sample(200)
        }
        self.assertEqual(severities, {"none", "moderate", "critical"})

    def test_every_sample_assembles_into_a_valid_event(self):
        for prediction in FakeSource(seed=12).sample(50):
            event = assemble_event(prediction).to_dict()
            self.assertEqual(event["mode"], "normal")
            self.assertTrue(event["typing_detected"])
            self.assertGreaterEqual(event["risk_score"], 0.0)
            self.assertLessEqual(event["risk_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
