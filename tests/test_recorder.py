import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from exposure.event import Prediction, assemble_event
from exposure.recorder import EventRecorder, next_free_path
from exposure.replay import ReplaySource
from exposure.source import FakeSource

CONFIDENT = [("p", 0.93), ("o", 0.03), ("l", 0.02), ("i", 0.01), ("k", 0.01)]


class TestNextFreePath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_unused_path_is_returned_unchanged(self):
        target = self.dir / "take.jsonl"
        self.assertEqual(next_free_path(target), target)

    def test_existing_path_gets_a_suffix(self):
        target = self.dir / "take.jsonl"
        target.write_text("x", encoding="utf-8")
        self.assertEqual(next_free_path(target).name, "take-1.jsonl")

    def test_suffix_increments_past_existing_variants(self):
        (self.dir / "take.jsonl").write_text("x", encoding="utf-8")
        (self.dir / "take-1.jsonl").write_text("x", encoding="utf-8")
        (self.dir / "take-2.jsonl").write_text("x", encoding="utf-8")
        self.assertEqual(next_free_path(self.dir / "take.jsonl").name, "take-3.jsonl")


class TestEventRecorder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_writes_one_line_per_event(self):
        path = self.dir / "take.jsonl"
        with EventRecorder(path) as recorder:
            for _ in range(3):
                recorder.write(assemble_event(Prediction(key_topk=CONFIDENT)).to_dict())
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)

    def test_counts_events(self):
        with EventRecorder(self.dir / "take.jsonl") as recorder:
            for _ in range(4):
                recorder.write(assemble_event(Prediction(key_topk=CONFIDENT)).to_dict())
            self.assertEqual(recorder.count, 4)

    def test_ignores_non_keystroke_messages(self):
        with EventRecorder(self.dir / "take.jsonl") as recorder:
            recorder.write({"type": "state", "latched": True})
            recorder.write(assemble_event(Prediction(key_topk=CONFIDENT)).to_dict())
            self.assertEqual(recorder.count, 1)

    def test_never_overwrites_an_existing_take(self):
        # A recorded session may be the only copy of a take that is hard to redo.
        first = self.dir / "take.jsonl"
        first.write_text("existing take\n", encoding="utf-8")
        recorder = EventRecorder(first)
        recorder.close()
        self.assertNotEqual(recorder.path, first)
        self.assertEqual(first.read_text(encoding="utf-8"), "existing take\n")

    def test_creates_missing_parent_directories(self):
        path = self.dir / "takes" / "nested" / "take.jsonl"
        with EventRecorder(path) as recorder:
            recorder.write(assemble_event(Prediction(key_topk=CONFIDENT)).to_dict())
        self.assertTrue(path.exists())

    def test_each_line_is_flushed_immediately(self):
        # An interrupted session must still leave a usable file.
        path = self.dir / "take.jsonl"
        recorder = EventRecorder(path)
        recorder.write(assemble_event(Prediction(key_topk=CONFIDENT)).to_dict())
        self.assertEqual(len(path.read_text(encoding="utf-8").strip().splitlines()), 1)
        recorder.close()

    def test_close_is_idempotent(self):
        recorder = EventRecorder(self.dir / "take.jsonl")
        recorder.close()
        recorder.close()

    def test_written_lines_are_valid_schema_json(self):
        path = self.dir / "take.jsonl"
        with EventRecorder(path) as recorder:
            recorder.write(assemble_event(Prediction(key_topk=CONFIDENT)).to_dict())
        record = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(record["type"], "keystroke")
        self.assertIn("risk_score", record)
        self.assertIn("alert_severity", record)


class TestRecordReplayRoundTrip(unittest.TestCase):
    """The workflow that matters: record a session, replay it identically."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_recorded_session_replays_with_identical_verdicts(self):
        path = self.dir / "take.jsonl"
        originals = [
            assemble_event(p) for p in FakeSource(seed=5).sample(25)
        ]
        with EventRecorder(path) as recorder:
            for event in originals:
                recorder.write(event.to_dict())

        source = ReplaySource(path, loop=False, speed=1000.0)

        async def drain():
            return [p async for p in source.predictions()]

        replayed = [assemble_event(p) for p in asyncio.run(drain())]

        self.assertEqual(len(replayed), len(originals))
        self.assertEqual(
            [e.alert_severity for e in replayed],
            [e.alert_severity for e in originals],
        )
        self.assertEqual(
            [e.key_top1 for e in replayed], [e.key_top1 for e in originals]
        )
        for replayed_event, original in zip(replayed, originals):
            self.assertAlmostEqual(replayed_event.risk_score, original.risk_score)

    def test_recorded_timing_survives_the_round_trip(self):
        path = self.dir / "take.jsonl"
        with EventRecorder(path) as recorder:
            for prediction in FakeSource(seed=9).sample(30):
                recorder.write(assemble_event(prediction).to_dict())
        source = ReplaySource(path)
        # Bursts and pauses both present, rather than one uniform gap.
        gaps = [g for g in source._gaps if g > 0]
        self.assertGreater(max(gaps), 1.0)
        self.assertLess(min(gaps), 0.3)


if __name__ == "__main__":
    unittest.main()
