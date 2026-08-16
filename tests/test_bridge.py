import unittest

from exposure.alert import CRITICAL_THRESHOLD, RISK_THRESHOLD
from exposure.bridge import TOP_K, normalize_label, prediction_from_pipeline_event
from exposure.event import assemble_event

# The classifier's real class set: 0-9, a-z, junk.
CLASSES = [str(d) for d in range(10)] + list("abcdefghijklmnopqrstuvwxyz") + ["junk"]


def pipeline_event(leader="e", leader_prob=0.9, **overrides):
    """Build an event shaped like pipeline.py's, with mass on `leader`."""
    rest = (1.0 - leader_prob) / (len(CLASSES) - 1)
    probs = [[c, leader_prob if c == leader else rest] for c in CLASSES]
    event = {
        "timestamp": 1786832104.5,
        "sample_index": 48000,
        "onset_strength": 3.2,
        "predicted_key": leader,
        "confidence": leader_prob,
        "key_probs": probs,
        "below_confidence_threshold": False,
        "is_junk": False,
        "snr_db": 18.2,
        "exposure_score": 64.3,
        "zone_breakdown": {},
        "model_trained": True,
    }
    event.update(overrides)
    return event


class TestNormalizeLabel(unittest.TestCase):
    def test_letters_lowercase(self):
        self.assertEqual(normalize_label("E"), "e")

    def test_space_aliases_map_to_a_literal_space(self):
        for alias in ("space", "SPACE", "spacebar", "_", "<space>"):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_label(alias), " ")

    def test_literal_space_passes_through(self):
        self.assertEqual(normalize_label(" "), " ")


class TestFiltering(unittest.TestCase):
    def test_junk_flag_drops_the_event(self):
        self.assertIsNone(prediction_from_pipeline_event(pipeline_event(is_junk=True)))

    def test_junk_as_argmax_drops_the_event(self):
        # Judged on the full distribution, before vocabulary filtering — otherwise
        # a junk sound would be reported as whichever letter came second.
        self.assertIsNone(prediction_from_pipeline_event(pipeline_event(leader="junk")))

    def test_low_confidence_drops_the_event(self):
        event = pipeline_event(below_confidence_threshold=True)
        self.assertIsNone(prediction_from_pipeline_event(event))

    def test_digit_prediction_drops_the_event(self):
        # A digit is outside the reportable set; promoting the best letter would
        # put a character in the transcript the model never predicted.
        self.assertIsNone(prediction_from_pipeline_event(pipeline_event(leader="7")))

    def test_missing_key_probs_drops_the_event(self):
        event = pipeline_event()
        del event["key_probs"]
        self.assertIsNone(prediction_from_pipeline_event(event))

    def test_empty_key_probs_drops_the_event(self):
        self.assertIsNone(prediction_from_pipeline_event(pipeline_event(key_probs=[])))

    def test_letter_prediction_is_kept(self):
        self.assertIsNotNone(prediction_from_pipeline_event(pipeline_event(leader="e")))

    def test_space_prediction_is_kept(self):
        # A model whose class set spells the space key "space" rather than " ".
        classes = [c for c in CLASSES if c != "junk"] + ["space"]
        rest = (1.0 - 0.9) / (len(classes) - 1)
        event = pipeline_event(leader="space")
        event["key_probs"] = [[c, 0.9 if c == "space" else rest] for c in classes]
        prediction = prediction_from_pipeline_event(event)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key_topk[0][0], " ")


class TestTranslation(unittest.TestCase):
    def test_key_topk_has_top_k_entries(self):
        prediction = prediction_from_pipeline_event(pipeline_event())
        self.assertEqual(len(prediction.key_topk), TOP_K)

    def test_key_topk_contains_only_reportable_characters(self):
        prediction = prediction_from_pipeline_event(pipeline_event())
        for char, _ in prediction.key_topk:
            self.assertTrue(char.islower() or char == " ", char)
            self.assertEqual(len(char), 1)

    def test_key_topk_is_sorted_and_normalized(self):
        prediction = prediction_from_pipeline_event(pipeline_event())
        probs = [p for _, p in prediction.key_topk]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_leader_survives_translation(self):
        prediction = prediction_from_pipeline_event(pipeline_event(leader="q"))
        self.assertEqual(prediction.key_topk[0][0], "q")

    def test_timestamp_is_carried_through(self):
        prediction = prediction_from_pipeline_event(pipeline_event())
        self.assertAlmostEqual(prediction.timestamp, 1786832104.5)

    def test_speech_present_is_false_since_the_pipeline_has_no_vad(self):
        prediction = prediction_from_pipeline_event(pipeline_event())
        self.assertFalse(prediction.speech_present)

    def test_probabilities_are_native_floats(self):
        prediction = prediction_from_pipeline_event(pipeline_event())
        for _, probability in prediction.key_topk:
            self.assertIs(type(probability), float)

    def test_accepts_a_dict_shaped_distribution(self):
        event = pipeline_event()
        event["key_probs"] = dict(event["key_probs"])
        self.assertIsNotNone(prediction_from_pipeline_event(event))

    def test_duplicate_labels_are_merged(self):
        # A retrain could emit two labels that normalize to the same character.
        event = pipeline_event(leader="e", leader_prob=0.5)
        event["key_probs"] = [["e", 0.5], ["E", 0.3], ["a", 0.2]]
        prediction = prediction_from_pipeline_event(event)
        self.assertEqual(prediction.key_topk[0][0], "e")
        self.assertAlmostEqual(prediction.key_topk[0][1], 0.8)

    def test_nan_probabilities_are_skipped(self):
        event = pipeline_event()
        event["key_probs"] = [["e", 0.9], ["a", float("nan")], ["b", 0.1]]
        prediction = prediction_from_pipeline_event(event)
        self.assertEqual([c for c, _ in prediction.key_topk], ["e", "b"])

    def test_unparseable_probabilities_are_skipped(self):
        event = pipeline_event()
        event["key_probs"] = [["e", 0.9], ["a", "not a number"], ["b", 0.1]]
        self.assertIsNotNone(prediction_from_pipeline_event(event))

    def test_single_surviving_candidate_is_dropped(self):
        # risk_score returns 0.0 for one candidate, so such an event could never
        # alert; emitting it would put a keystroke on screen invisible to alerts.
        event = pipeline_event()
        event["key_probs"] = [["e", 1.0], ["junk", 0.0]]
        self.assertIsNone(prediction_from_pipeline_event(event))


class TestEndToEnd(unittest.TestCase):
    """Pipeline event → Prediction → assembled schema event."""

    def test_confident_prediction_produces_a_critical_alert(self):
        event = assemble_event(prediction_from_pipeline_event(pipeline_event(leader_prob=0.995)))
        self.assertEqual(event.key_top1, "e")
        self.assertGreater(event.risk_score, CRITICAL_THRESHOLD)
        self.assertEqual(event.alert_severity, "critical")

    def test_ambiguous_prediction_does_not_alert(self):
        event = assemble_event(prediction_from_pipeline_event(pipeline_event(leader_prob=0.2)))
        self.assertLess(event.risk_score, RISK_THRESHOLD)
        self.assertFalse(event.alert)

    def test_result_serializes_to_the_frozen_schema(self):
        payload = assemble_event(prediction_from_pipeline_event(pipeline_event())).to_dict()
        self.assertEqual(payload["type"], "keystroke")
        self.assertEqual(payload["mode"], "normal")
        self.assertTrue(payload["typing_detected"])
        self.assertIsInstance(payload["risk_score"], float)

    def test_result_is_json_serializable(self):
        import json

        payload = assemble_event(prediction_from_pipeline_event(pipeline_event())).to_dict()
        json.loads(json.dumps(payload))  # would raise on numpy scalars

    def test_risk_rises_monotonically_with_model_confidence(self):
        scores = [
            assemble_event(prediction_from_pipeline_event(pipeline_event(leader_prob=p))).risk_score
            for p in (0.2, 0.5, 0.8, 0.95, 0.995)
        ]
        self.assertEqual(scores, sorted(scores))


if __name__ == "__main__":
    unittest.main()
