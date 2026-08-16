"""
Readable console output for the live pipeline -- replaces printing raw
event dicts. Collapses repetitive "junk" events into a running counter
instead of spamming one line per non-keystroke sound, and shows real
predictions prominently.

Usage (drop-in replacement for a bare `print(event)` on_event callback):

    from console_display import ConsoleDisplay
    display = ConsoleDisplay()
    pipeline = AcousticGuardPipeline(on_event=display.handle_event, checkpoint_path=...)
"""

from __future__ import annotations
import time


class ConsoleDisplay:
    def __init__(self, junk_summary_every: int = 10):
        self.junk_count = 0
        self.low_conf_count = 0
        self.key_count = 0
        self.key_tally: dict[str, int] = {}
        self.junk_summary_every = junk_summary_every
        self._session_start = time.time()

    def handle_event(self, event: dict) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))

        if event["is_junk"]:
            self.junk_count += 1
            if self.junk_count % self.junk_summary_every == 0:
                print(f"[{ts}] ({self.junk_count} background/non-keystroke sounds filtered so far)")
            return

        if event["below_confidence_threshold"]:
            self.low_conf_count += 1
            print(f"[{ts}] uncertain -- possible '{event['predicted_key']}' "
                  f"but only {event['confidence']*100:.0f}% confident (below threshold, not counted)")
            return

        # a real, confident, non-junk prediction
        self.key_count += 1
        key = event["predicted_key"]
        self.key_tally[key] = self.key_tally.get(key, 0) + 1
        print(f"[{ts}] KEY: '{key}'   confidence: {event['confidence']*100:.0f}%   "
              f"exposure score: {event['exposure_score']:.1f}")

    def print_summary(self) -> None:
        elapsed = time.time() - self._session_start
        print("\n" + "=" * 50)
        print(f"Session summary ({elapsed:.0f}s):")
        print(f"  Keystrokes detected : {self.key_count}")
        print(f"  Uncertain/low-conf  : {self.low_conf_count}")
        print(f"  Background/junk     : {self.junk_count}")
        if self.key_tally:
            top = sorted(self.key_tally.items(), key=lambda kv: -kv[1])[:10]
            top_str = ", ".join(f"'{k}'x{c}" for k, c in top)
            print(f"  Most common keys    : {top_str}")
        print("=" * 50)


class GroundTruthTester:
    """Optional: type a known string and compare live predictions against it,
    for a real accuracy readout during testing (rather than just watching
    predictions go by with no ground truth to check them against).

    Usage:
        tester = GroundTruthTester(expected_text="hello world")
        pipeline = AcousticGuardPipeline(on_event=tester.handle_event, checkpoint_path=...)
        # ... type "hello world" on the capture device ...
        tester.print_report()

    NOTE: this assumes keystrokes arrive in the same order as expected_text
    and only counts non-junk, above-threshold predictions -- it will get
    thrown off by extra/missed detections shifting the alignment, since it
    does simple positional comparison rather than sequence alignment. Good
    enough for a quick accuracy sanity check, not a rigorous eval.
    """

    def __init__(self, expected_text: str):
        self.expected = list(expected_text.replace(" ", ""))  # space isn't in accepted_keys
        self.position = 0
        self.correct = 0
        self.total = 0
        self.log: list[tuple[str, str, bool]] = []  # (expected, predicted, correct)

    def handle_event(self, event: dict) -> None:
        if event["is_junk"] or event["below_confidence_threshold"]:
            return
        if self.position >= len(self.expected):
            return  # typed more than expected -- ignore extras

        expected_char = self.expected[self.position]
        predicted_char = event["predicted_key"]
        is_correct = predicted_char == expected_char
        self.log.append((expected_char, predicted_char, is_correct))
        self.total += 1
        if is_correct:
            self.correct += 1
        self.position += 1

        mark = "✓" if is_correct else "✗"
        print(f"  [{mark}] expected '{expected_char}' -> predicted '{predicted_char}' "
              f"({event['confidence']*100:.0f}%)")

    def print_report(self) -> None:
        acc = (self.correct / self.total * 100) if self.total else 0.0
        print("\n" + "=" * 50)
        print(f"Ground-truth accuracy: {self.correct}/{self.total} ({acc:.1f}%)")
        if self.total < len(self.expected):
            print(f"  (only {self.total}/{len(self.expected)} expected characters were detected at all)")
        wrong = [(e, p) for e, p, c in self.log if not c]
        if wrong:
            print(f"  Mistakes: {', '.join(f'{e}->{p}' for e, p in wrong)}")
        print("=" * 50)
