import time
import unittest

from exposure.event import Prediction, assemble_event

SCHEMA_FIELDS = [
    "type",
    "key_top1",
    "key_topk",
    "confidence",
    "timestamp",
    "mode",
    "risk_score",
    "typing_detected",
    "speech_present",
    "alert",
    "alert_severity",
]

CONFIDENT = [("p", 0.93), ("o", 0.03), ("l", 0.02), ("i", 0.01), ("k", 0.01)]
AMBIGUOUS = [("p", 0.22), ("o", 0.21), ("l", 0.20), ("i", 0.19), ("k", 0.18)]


class TestEventAssembly(unittest.TestCase):
    def test_emits_exactly_the_frozen_schema(self):
        event = assemble_event(Prediction(key_topk=CONFIDENT)).to_dict()
        self.assertEqual(list(event.keys()), SCHEMA_FIELDS)

    def test_type_is_keystroke(self):
        self.assertEqual(assemble_event(Prediction(key_topk=CONFIDENT)).type, "keystroke")

    def test_mode_is_always_normal(self):
        # No context signal, no password detection in this version.
        for topk in (CONFIDENT, AMBIGUOUS):
            with self.subTest(topk=topk):
                self.assertEqual(assemble_event(Prediction(key_topk=topk)).mode, "normal")

    def test_key_top1_is_leader(self):
        self.assertEqual(assemble_event(Prediction(key_topk=CONFIDENT)).key_top1, "p")

    def test_confidence_defaults_to_leader_probability(self):
        self.assertAlmostEqual(
            assemble_event(Prediction(key_topk=CONFIDENT)).confidence, 0.93
        )

    def test_explicit_confidence_is_preserved(self):
        prediction = Prediction(key_topk=CONFIDENT, confidence=0.5)
        self.assertAlmostEqual(assemble_event(prediction).confidence, 0.5)

    def test_timestamp_defaults_to_now(self):
        before = time.time()
        event = assemble_event(Prediction(key_topk=CONFIDENT))
        self.assertGreaterEqual(event.timestamp, before)
        self.assertLessEqual(event.timestamp, time.time())

    def test_explicit_timestamp_is_preserved(self):
        prediction = Prediction(key_topk=CONFIDENT, timestamp=1234.5)
        self.assertEqual(assemble_event(prediction).timestamp, 1234.5)

    def test_speech_present_passes_through(self):
        for speech in (True, False):
            with self.subTest(speech=speech):
                prediction = Prediction(key_topk=CONFIDENT, speech_present=speech)
                self.assertEqual(assemble_event(prediction).speech_present, speech)

    def test_speech_present_does_not_change_the_alert(self):
        # The severity table is explicit that speech never suppresses an alert.
        loud = assemble_event(Prediction(key_topk=CONFIDENT, speech_present=True))
        quiet = assemble_event(Prediction(key_topk=CONFIDENT, speech_present=False))
        self.assertEqual(loud.alert, quiet.alert)
        self.assertEqual(loud.alert_severity, quiet.alert_severity)
        self.assertAlmostEqual(loud.risk_score, quiet.risk_score)

    def test_confident_prediction_raises_an_alert(self):
        event = assemble_event(Prediction(key_topk=CONFIDENT))
        self.assertTrue(event.alert)
        self.assertIn(event.alert_severity, ("moderate", "critical"))

    def test_ambiguous_prediction_does_not_alert(self):
        event = assemble_event(Prediction(key_topk=AMBIGUOUS))
        self.assertFalse(event.alert)
        self.assertEqual(event.alert_severity, "none")

    def test_empty_prediction_is_not_typing(self):
        event = assemble_event(Prediction(key_topk=[]))
        self.assertFalse(event.typing_detected)
        self.assertFalse(event.alert)
        self.assertEqual(event.key_top1, "")

    def test_risk_ignores_key_identity(self):
        # The architectural rule, tested: relabelling the keys must not move the risk
        # score. If this ever fails, the transcript branch has leaked into the alert
        # branch.
        original = assemble_event(Prediction(key_topk=CONFIDENT))
        relabelled = assemble_event(
            Prediction(key_topk=[("z", p) for _, p in CONFIDENT])
        )
        self.assertAlmostEqual(original.risk_score, relabelled.risk_score)

    def test_topk_serializes_as_lists_not_tuples(self):
        event = assemble_event(Prediction(key_topk=CONFIDENT)).to_dict()
        self.assertTrue(all(isinstance(pair, list) for pair in event["key_topk"]))


if __name__ == "__main__":
    unittest.main()
