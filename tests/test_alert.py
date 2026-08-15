import unittest

from exposure.alert import CRITICAL_THRESHOLD, RISK_THRESHOLD, decide_alert


class TestAlertDecision(unittest.TestCase):
    def test_no_typing_never_alerts(self):
        for risk in (0.0, 0.5, 0.99, 1.0):
            with self.subTest(risk=risk):
                self.assertEqual(decide_alert(risk, typing_detected=False), (False, "none"))

    def test_below_threshold_is_log_only(self):
        self.assertEqual(decide_alert(0.1, typing_detected=True), (False, "none"))

    def test_above_risk_threshold_is_moderate(self):
        alert, severity = decide_alert(RISK_THRESHOLD + 0.01, typing_detected=True)
        self.assertTrue(alert)
        self.assertEqual(severity, "moderate")

    def test_above_critical_threshold_is_critical(self):
        alert, severity = decide_alert(CRITICAL_THRESHOLD + 0.01, typing_detected=True)
        self.assertTrue(alert)
        self.assertEqual(severity, "critical")

    def test_thresholds_are_exclusive(self):
        # A score sitting exactly on a threshold must not fire, so a default-valued
        # score cannot trip an alert by coincidence.
        self.assertEqual(decide_alert(RISK_THRESHOLD, typing_detected=True), (False, "none"))
        _, severity = decide_alert(CRITICAL_THRESHOLD, typing_detected=True)
        self.assertEqual(severity, "moderate")

    def test_severity_ordering_across_the_range(self):
        observed = [
            decide_alert(risk, typing_detected=True)[1]
            for risk in (0.0, 0.6, 0.9)
        ]
        self.assertEqual(observed, ["none", "moderate", "critical"])

    def test_thresholds_are_injectable_for_calibration(self):
        alert, severity = decide_alert(
            0.3, typing_detected=True, risk_threshold=0.2, critical_threshold=0.9
        )
        self.assertTrue(alert)
        self.assertEqual(severity, "moderate")

    def test_alert_flag_agrees_with_severity(self):
        for risk in (0.0, 0.3, 0.56, 0.7, 0.81, 1.0):
            with self.subTest(risk=risk):
                alert, severity = decide_alert(risk, typing_detected=True)
                self.assertEqual(alert, severity != "none")


if __name__ == "__main__":
    unittest.main()
