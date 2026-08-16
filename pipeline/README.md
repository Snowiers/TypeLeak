+6b22222hjhhhhzzxzzzs# Acoustic Guard — Streaming Pipeline

Real-time audio → onset detection → mel-spectrogram → CNN zone classifier →
rolling exposure score. See the project doc for the full system design and
rubric framing; this is the implementation of the core pipeline.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunable constants in one place. **Start here** when tuning for your mic/room. |
| `audio_io.py` | `RingBuffer` (thread-safe circular audio buffer), `MicAudioSource` (live mic via `sounddevice`), `ArrayAudioSource` (replays a numpy array — for testing and semi-live demo fallback). |
| `onset_detector.py` | Spectral-flux based keystroke onset detection, streaming/incremental. |
| `features.py` | Log-mel spectrogram extraction (torchaudio) + SNR estimate per event. |
| `model.py` | `KeystrokeZoneCNN` (small conv net) + `ZoneClassifier` inference wrapper. Works untrained out of the box. |
| `exposure.py` | `NoiseFloorTracker` (ambient RMS estimate) + `ExposureScorer` (rolling 0-100 score, time+SNR weighted). |
| `pipeline.py` | `AcousticGuardPipeline` — wires everything together. **This is what you import and use.** |
| `test_offline.py` | No-hardware smoke test with synthetic clicks. Run this first, and after any change. |

## Quick start

```bash
pip install torch torchaudio numpy scipy sounddevice --break-system-packages
python3 test_offline.py
```

You should see 5/5 synthetic clicks detected with no false positives. If you
change onset detection parameters, **re-run this** — it's your regression test.

## Running live (on the demo machine, with a real mic)

```python
from pipeline import AcousticGuardPipeline
from audio_io import MicAudioSource

def on_event(event):
    print(event)
    # event = {
    #   "timestamp": ..., "sample_index": ..., "onset_strength": ...,
    #   "predicted_zone": "home_row", "zone_confidence": 0.83,
    #   "snr_db": 18.2, "exposure_score": 64.3,
    #   "zone_breakdown": {"home_row": 61.0, "left_hand": 70.2, ...},
    #   "model_trained": True/False
    # }
    # -> feed this into your dashboard UI here.

pipeline = AcousticGuardPipeline(on_event=on_event)
pipeline.run(MicAudioSource())   # blocks; Ctrl+C to stop
```

## What's NOT built yet (next steps for the team)

1. **Training script.** `model.py` defines the architecture and inference
   wrapper, but there's no `train.py` yet. Plan:
   - Load JBFH-Dev/Keystroke-Datasets (per-key .wav clips), map each key to
     one of `config.ZONE_LABELS`.
   - Reuse `MelFeatureExtractor` to build training features consistently
     with what the live pipeline produces (important — don't build a
     separate offline feature pipeline that could drift from this one).
   - Standard train/val split, cross-entropy loss, save with
     `torch.save(model.state_dict(), path)` .
   - Point `config.MODEL_CHECKPOINT_PATH` at the saved file, or pass
     `checkpoint_path=...` to `AcousticGuardPipeline`.
   - **Record supplementary data on your actual demo keyboard/mic** —
     accuracy is device-specific (switch type, mic placement), so the
     public dataset alone likely won't transfer perfectly. Even 10-15 min
     of real data on your demo hardware should help a lot.

2. **Tune onset detection on real audio, not synthetic.** The default
   `ONSET_THRESHOLD_K = 6.0` was tuned against synthetic Gaussian noise +
   clean clicks (see `test_offline.py`). Real rooms have structured noise
   (HVAC hum, voices, etc.) with a different spectral-flux profile — record
   a few minutes of real typing + real background noise and re-tune `K`
   (and possibly `ONSET_REFRACTORY_MS`) against that before the demo.

3. **TensorRT export**, once you have a trained model — this is your
   concrete "before/after" latency number for the hardware-proof segment
   of the demo video.

4. **Dashboard.** `pipeline.run()` calls `on_event(dict)` for every detected
   keystroke — that's the integration point. A simple approach: have
   `on_event` push into a queue and serve it over a local websocket to a
   small web UI (waveform + live exposure score + zone heatmap).

5. **Remediation toggle demo.** For the "score visibly drops" moment in the
   video: the cleanest way is probably to actually enable real OS-level or
   virtual-mic noise suppression and let `NoiseFloorTracker`/SNR naturally
   reflect it — rather than faking the score — since it's more convincing
   and it's not much extra work given the SNR-weighting is already in
   `exposure.py`.

## Known simplifications (be ready to explain these to judges if asked)

- Zone labels are coarse by design (privacy framing), not a technical
  ceiling — published research achieves full per-key accuracy on similar
  pipelines.
- Onset detection currently only fires once per keystroke via the
  refractory period; it doesn't yet distinguish press vs. release
  transients (some published attacks use both — a possible stretch goal).
- `ExposureScorer`'s weighting formula (time decay × SNR factor) is a
  reasonable first design, not a validated metric — if you have time,
  it's worth sanity-checking the weights against a couple of real
  recorded sessions (quiet room vs. noisy room) to make sure the score
  actually moves in the direction you'd expect.
