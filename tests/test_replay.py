import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from exposure.event import assemble_event
from exposure.replay import DEFAULT_GAP_SECONDS, ReplaySource
from exposure.source import FakeSource


def write_fixture(events, directory):
    path = Path(directory) / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return path


def sample_events(count=6, seed=1):
    return [assemble_event(p).to_dict() for p in FakeSource(seed=seed).sample(count)]


async def take(source, count):
    collected = []
    async for prediction in source.predictions():
        collected.append(prediction)
        if len(collected) == count:
            break
    return collected


class TestReplaySource(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def test_loads_all_keystroke_events(self):
        path = write_fixture(sample_events(6), self.dir)
        self.assertEqual(ReplaySource(path).length, 6)

    def test_ignores_non_keystroke_records(self):
        events = sample_events(4) + [{"type": "state", "latched": True}]
        path = write_fixture(events, self.dir)
        self.assertEqual(ReplaySource(path).length, 4)

    def test_ignores_blank_lines(self):
        path = Path(self.dir) / "events.jsonl"
        body = "\n".join(json.dumps(e) for e in sample_events(3))
        path.write_text(f"\n{body}\n\n", encoding="utf-8")
        self.assertEqual(ReplaySource(path).length, 3)

    def test_rejects_malformed_json_with_line_number(self):
        path = Path(self.dir) / "bad.jsonl"
        good = json.dumps(sample_events(1)[0])
        path.write_text(f"{good}\nnot json\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            ReplaySource(path)
        self.assertIn(":2:", str(ctx.exception))

    def test_rejects_keystroke_record_missing_key_topk(self):
        path = write_fixture([{"type": "keystroke", "confidence": 0.9}], self.dir)
        with self.assertRaises(ValueError) as ctx:
            ReplaySource(path)
        self.assertIn("key_topk", str(ctx.exception))

    def test_rejects_file_with_no_keystrokes(self):
        path = write_fixture([{"type": "state"}], self.dir)
        with self.assertRaises(ValueError):
            ReplaySource(path)

    def test_replays_in_recorded_order(self):
        events = sample_events(5)
        path = write_fixture(events, self.dir)
        source = ReplaySource(path, loop=False, speed=1000.0)
        replayed = asyncio.run(take(source, 5))
        self.assertEqual(
            [p.key_topk[0][0] for p in replayed],
            [e["key_top1"] for e in events],
        )

    def test_preserves_prediction_fields(self):
        events = sample_events(3)
        path = write_fixture(events, self.dir)
        source = ReplaySource(path, loop=False, speed=1000.0)
        replayed = asyncio.run(take(source, 3))
        for original, prediction in zip(events, replayed):
            self.assertAlmostEqual(prediction.confidence, original["confidence"])
            self.assertEqual(prediction.speech_present, original["speech_present"])
            self.assertEqual(len(prediction.key_topk), len(original["key_topk"]))

    def test_restamps_timestamps_by_default(self):
        # A replayed stream showing yesterday's clock times looks broken on stage.
        events = sample_events(3)
        path = write_fixture(events, self.dir)
        source = ReplaySource(path, loop=False, speed=1000.0)
        replayed = asyncio.run(take(source, 3))
        self.assertGreater(replayed[0].timestamp, events[0]["timestamp"])

    def test_can_preserve_original_timestamps(self):
        events = sample_events(3)
        path = write_fixture(events, self.dir)
        source = ReplaySource(path, loop=False, speed=1000.0, preserve_timestamps=True)
        replayed = asyncio.run(take(source, 3))
        self.assertAlmostEqual(replayed[0].timestamp, events[0]["timestamp"])

    def test_loops_past_the_end(self):
        events = sample_events(3)
        path = write_fixture(events, self.dir)
        source = ReplaySource(path, loop=True, speed=1000.0)
        replayed = asyncio.run(take(source, 7))
        self.assertEqual(len(replayed), 7)
        self.assertEqual(replayed[0].key_topk[0][0], replayed[3].key_topk[0][0])

    def test_stops_when_not_looping(self):
        path = write_fixture(sample_events(3), self.dir)
        source = ReplaySource(path, loop=False, speed=1000.0)
        replayed = asyncio.run(take(source, 99))
        self.assertEqual(len(replayed), 3)

    def test_long_gaps_are_compressed(self):
        events = sample_events(2)
        events[0]["timestamp"] = 1000.0
        events[1]["timestamp"] = 1600.0  # ten minutes later
        path = write_fixture(events, self.dir)
        source = ReplaySource(path, max_gap=2.0)
        self.assertLessEqual(max(source._gaps), 2.0)

    def test_degenerate_timing_falls_back_to_default_gap(self):
        # The failure mode this guards: a fixture dumped in a tight loop, where every
        # timestamp is identical and naive replay would fire everything at once.
        events = sample_events(4)
        for event in events:
            event["timestamp"] = 500.0
        path = write_fixture(events, self.dir)
        source = ReplaySource(path)
        self.assertTrue(all(g == 0.0 or g == DEFAULT_GAP_SECONDS for g in source._gaps))

    def test_replayed_predictions_reassemble_into_events(self):
        # Replay runs through the live risk/alert path, so a threshold change shows up
        # in replay rather than being frozen into the recording.
        path = write_fixture(sample_events(5), self.dir)
        source = ReplaySource(path, loop=False, speed=1000.0)
        for prediction in asyncio.run(take(source, 5)):
            event = assemble_event(prediction).to_dict()
            self.assertEqual(event["type"], "keystroke")
            self.assertIn(event["alert_severity"], ("none", "moderate", "critical"))

    def test_replay_reproduces_original_severities(self):
        events = sample_events(20)
        path = write_fixture(events, self.dir)
        source = ReplaySource(path, loop=False, speed=1000.0)
        replayed = [assemble_event(p) for p in asyncio.run(take(source, 20))]
        self.assertEqual(
            [e.alert_severity for e in replayed],
            [e["alert_severity"] for e in events],
        )


if __name__ == "__main__":
    unittest.main()
