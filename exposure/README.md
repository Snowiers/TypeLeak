# exposure — risk, alert decision, transport

The middle of the pipeline. Consumes a `Prediction` from the model side, produces the
frozen audio event schema over a local websocket. No audio libraries, no model
dependencies, no OS APIs, no frontend code.

Observation only — nothing here modifies the audio stream.

## Run it

```
pip install -r requirements.txt

python -m exposure                    # fake events on ws://localhost:8765
python -m exposure --source replay    # deterministic playback, for rehearsal
python -m exposure --dump 40          # print 40 events as JSONL, no websockets needed
python -m exposure --list-sources     # what is available and when to use it
python -m unittest discover -s tests
```

`fixtures/sample_events.jsonl` is 200 pre-generated events spanning about 65 seconds of
realistic typing rhythm, committed so the frontend can be built against a static file
with no Python running at all.

### Sources, and the degradation switch

| `--source` | Behaviour |
|---|---|
| `fake` | Randomised. Different every run. Development default |
| `replay` | Deterministic playback of a JSONL file. Same alerts, same moments, every run |
| `classifier` | The real model side. Not wired up yet |

`replay` is demo insurance. Rehearsing against `fake` means practising over a stream
you have never seen; `replay` lets you learn where the alerts land. When the real
classifier is wired in, falling back mid-demo is a flag rather than a code edit.

Useful replay options: `--fixture PATH`, `--speed 2.0`, `--no-loop`.

Replay reconstructs upstream predictions and runs them through the live risk and alert
code, rather than replaying frozen verdicts. A threshold change is therefore visible in
replay instead of being baked into the recording.

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

Two ready-made adapters in `classifier.py`, both already tested. Pick whichever fits
the shape of your code; nothing downstream changes either way.

**You push** (best fit if you own a running capture loop):

```python
from exposure.classifier import QueueSource
from exposure.server import EventServer

source = QueueSource()
server = EventServer(source)          # instead of FakeSource()

# from your audio loop, any thread:
source.submit(Prediction(key_topk=topk, speech_present=vad_flag))
```

**We pull** (best fit if your code is request-shaped):

```python
from exposure.classifier import CallableSource

source = CallableSource(your_async_function)   # returns Prediction, or None to stop
```

`QueueSource` bounds its backlog at 256 and drops the oldest prediction under
pressure — for a live exposure monitor a stale keystroke is worth less than a current
one, and an unbounded queue would eventually take the process down mid-demo. Check
`source.dropped` if throughput looks suspicious.

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
| `replay.py` | Deterministic playback from a JSONL file |
| `classifier.py` | `QueueSource` and `CallableSource` — where the model side plugs in |
| `state.py` | Latch and bounded alert log |
| `server.py` | Websocket transport |

## The architectural rule

`risk_score()` takes a sequence of probabilities and nothing else. It has no parameter
that could carry a key identity or decoded text, so the alert branch cannot come to
depend on the transcript branch through a later edit. `test_risk_ignores_key_identity`
in `tests/test_event.py` pins this: relabelling every key must leave the risk score
unchanged. If that test ever fails, the separation has broken.
