# exposure — risk, alert decision, transport

The middle of the pipeline. Consumes a `Prediction` from the model side, produces the
frozen audio event schema over a local websocket. No audio libraries, no model
dependencies, no OS APIs, no frontend code.

Observation only — nothing here modifies the audio stream.

## Run it

```
pip install -r requirements.txt

python -m exposure                 # serve fake events on ws://localhost:8765
python -m exposure --dump 40       # print 40 events as JSONL, no websockets needed
python -m exposure --seed 42       # reproducible fake stream
python -m unittest discover -s tests
```

`fixtures/sample_events.jsonl` is 60 pre-generated events, committed so the frontend
can be built against a static file with no Python running at all.

## For the frontend

Connect to `ws://localhost:8765`. Two message types arrive, distinguished by `type`.

**`type: "keystroke"`** — one per detected keystroke, the frozen schema:

```json
{
  "type": "keystroke",
  "key_top1": "e",
  "key_topk": [["e", 0.94], ["o", 0.02], ["t", 0.02], ["g", 0.01], ["a", 0.01]],
  "confidence": 0.94,
  "timestamp": 1786832104.6256788,
  "mode": "normal",
  "risk_score": 0.86,
  "typing_detected": true,
  "speech_present": false,
  "alert": true,
  "alert_severity": "critical"
}
```

Notes that will save you time:

- `confidence` is the classifier's top-1 probability. `risk_score` is the independent
  exposure estimate and is what drives the alert. Different numbers, don't conflate
  them in the UI.
- `mode` is always `"normal"` in this version. Show the label anyway — it's the
  transcript panel's mode indicator, and a later version varies it.
- `speech_present` is `false` when the model side doesn't surface VAD output. Label
  that case unknown rather than rendering it as confirmed silence.
- `alert_severity` is `"none"`, `"moderate"`, or `"critical"`. Thresholds are
  placeholders until calibration, so expect the mix to shift.

**`type: "state"`** — sent on connect, and again after a latch clear:

```json
{
  "type": "state",
  "latched": true,
  "peak_severity": "critical",
  "total_events": 128,
  "total_alerts": 34,
  "recent_alerts": [ ... keystroke events, newest first ... ]
}
```

This is not the frozen schema — it's a state summary so a reloaded page rejoins
showing the alerts it missed instead of an empty log.

**The exposure indicator latches.** No event is emitted when typing stops, so there's
no "typing ended" signal to switch it off. It stays lit until cleared. On
`Ctrl+Shift+X` (window-focus only, no global hotkey), send:

```json
{"type": "clear_latch"}
```

A fresh `state` message comes back with `latched: false`. The alert log survives the
clear.

## For the model side

Hand over one `Prediction` per detected keystroke:

```python
from exposure.event import Prediction

Prediction(
    key_topk=[("e", 0.94), ("o", 0.02), ("t", 0.02), ("g", 0.01), ("a", 0.01)],
    confidence=0.94,        # optional, defaults to key_topk[0][1]
    timestamp=onset_time,   # optional, defaults to arrival time
    speech_present=vad_flag,
)
```

Then implement `EventSource` — an async iterator yielding those — and construct
`EventServer` with it instead of `FakeSource`. Nothing else changes.

`timestamp` should be stamped at onset detection, not at handoff. Arrival time drifts
by however long inference took, which is exactly the latency the demo is about.

`key_topk` probabilities need not be normalized; the risk function normalizes them.

## Layout

| File | What |
|---|---|
| `risk.py` | `key_topk` probabilities → risk score. Pure function |
| `alert.py` | Risk + typing → `alert`, `alert_severity`. The two tuning constants live here |
| `event.py` | Both seams: `Prediction` in, `AudioEvent` out |
| `source.py` | `EventSource` interface plus `FakeSource` |
| `state.py` | Latch and bounded alert log |
| `server.py` | Websocket transport |

## The architectural rule

`risk_score()` takes a sequence of probabilities and nothing else. It has no parameter
that could carry a key identity or decoded text, so the alert branch cannot come to
depend on the transcript branch through a later edit. `test_risk_ignores_key_identity`
in `tests/test_event.py` pins this: relabelling every key must leave the risk score
unchanged. If that test ever fails, the separation has broken.
