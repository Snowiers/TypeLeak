import unittest

from exposure.event import Prediction, assemble_event
from exposure.state import ExposureState

# Chosen to land in each severity band at the 0.80 / 0.95 thresholds in alert.py.
# test_severity_fixtures_land_in_their_bands below pins that, so a recalibration
# fails loudly here rather than silently weakening every test in this file.
CONFIDENT = [("p", 0.995), ("o", 0.002), ("l", 0.001), ("i", 0.001), ("k", 0.001)]
MODERATE = [("a", 0.93), ("s", 0.03), ("d", 0.02), ("f", 0.01), ("g", 0.01)]
AMBIGUOUS = [("p", 0.22), ("o", 0.21), ("l", 0.20), ("i", 0.19), ("k", 0.18)]


def event(topk):
    return assemble_event(Prediction(key_topk=topk))


class TestSeverityFixtures(unittest.TestCase):
    """Guards the fixtures above against a threshold change."""

    def test_severity_fixtures_land_in_their_bands(self):
        self.assertEqual(event(CONFIDENT).alert_severity, "critical")
        self.assertEqual(event(MODERATE).alert_severity, "moderate")
        self.assertEqual(event(AMBIGUOUS).alert_severity, "none")


class TestExposureState(unittest.TestCase):
    def setUp(self):
        self.state = ExposureState()

    def test_starts_unlatched(self):
        self.assertFalse(self.state.latched)
        self.assertEqual(self.state.peak_severity, "none")

    def test_non_alert_event_does_not_latch(self):
        self.state.record(event(AMBIGUOUS))
        self.assertFalse(self.state.latched)
        self.assertEqual(self.state.recent_alerts(), [])

    def test_alert_latches(self):
        self.state.record(event(CONFIDENT))
        self.assertTrue(self.state.latched)

    def test_latch_survives_subsequent_quiet_events(self):
        # The core reason the latch exists: nothing is emitted when typing stops, so
        # quiet must never silently clear the indicator.
        self.state.record(event(CONFIDENT))
        for _ in range(5):
            self.state.record(event(AMBIGUOUS))
        self.assertTrue(self.state.latched)

    def test_manual_clear_resets_latch_and_peak(self):
        self.state.record(event(CONFIDENT))
        self.state.clear_latch()
        self.assertFalse(self.state.latched)
        self.assertEqual(self.state.peak_severity, "none")

    def test_manual_clear_keeps_the_log(self):
        self.state.record(event(CONFIDENT))
        self.state.clear_latch()
        self.assertEqual(len(self.state.recent_alerts()), 1)

    def test_peak_severity_holds_the_maximum(self):
        self.state.record(event(CONFIDENT))
        self.state.record(event(MODERATE))
        self.assertEqual(self.state.peak_severity, "critical")

    def test_peak_severity_climbs(self):
        self.state.record(event(MODERATE))
        self.assertEqual(self.state.peak_severity, "moderate")
        self.state.record(event(CONFIDENT))
        self.assertEqual(self.state.peak_severity, "critical")

    def test_recent_alerts_are_newest_first(self):
        first = event(CONFIDENT)
        second = event(MODERATE)
        self.state.record(first)
        self.state.record(second)
        self.assertEqual(self.state.recent_alerts()[0].key_top1, second.key_top1)

    def test_recent_alerts_respects_limit(self):
        for _ in range(5):
            self.state.record(event(CONFIDENT))
        self.assertEqual(len(self.state.recent_alerts(limit=2)), 2)

    def test_log_is_bounded(self):
        state = ExposureState(log_size=3)
        for _ in range(10):
            state.record(event(CONFIDENT))
        self.assertEqual(len(state.recent_alerts()), 3)

    def test_snapshot_shape(self):
        self.state.record(event(CONFIDENT))
        self.state.record(event(AMBIGUOUS))
        snapshot = self.state.snapshot()
        self.assertEqual(snapshot["type"], "state")
        self.assertTrue(snapshot["latched"])
        self.assertEqual(snapshot["peak_severity"], "critical")
        self.assertEqual(snapshot["total_events"], 2)
        self.assertEqual(snapshot["total_alerts"], 1)
        self.assertEqual(len(snapshot["recent_alerts"]), 1)

    def test_snapshot_alerts_are_serializable_dicts(self):
        self.state.record(event(CONFIDENT))
        alert = self.state.snapshot()["recent_alerts"][0]
        self.assertIsInstance(alert, dict)
        self.assertEqual(alert["type"], "keystroke")


if __name__ == "__main__":
    unittest.main()
