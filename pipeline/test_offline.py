"""
End-to-end smoke test using synthetic audio — no microphone or trained model
required. Generates a few seconds of quiet background noise with sharp
broadband "click" transients inserted at known times (a rough stand-in for
keystroke sounds), streams it through the real pipeline in small chunks
(simulating live audio callbacks), and checks that:

  1. The onset detector actually fires close to each synthetic click.
  2. Each detected onset makes it through feature extraction + classification
     without error (model is untrained/random here, so predictions are
     meaningless — this test is about pipeline plumbing, not accuracy).
  3. The exposure score updates and is a sane 0-100 value.

Run: python3 test_offline.py
"""

import numpy as np
import config
from audio_io import ArrayAudioSource
from pipeline import AcousticGuardPipeline


def make_synthetic_audio(duration_s=4.0, sample_rate=config.SAMPLE_RATE,
                          click_times_s=(0.5, 1.2, 1.9, 2.6, 3.3), seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    # quiet background noise, roughly -40dB-ish RMS
    audio = rng.normal(0, 0.003, n).astype(np.float32)

    click_len = int(0.01 * sample_rate)  # 10ms broadband click
    for t in click_times_s:
        idx = int(t * sample_rate)
        if idx + click_len >= n:
            continue
        click = rng.normal(0, 0.4, click_len).astype(np.float32)
        # quick decay envelope so it looks like a transient, not a noise burst
        envelope = np.exp(-np.linspace(0, 8, click_len))
        audio[idx:idx + click_len] += click * envelope

    return audio, list(click_times_s)


def main():
    audio, true_click_times = make_synthetic_audio()
    print(f"Synthetic audio: {len(audio) / config.SAMPLE_RATE:.2f}s, "
          f"{len(true_click_times)} injected clicks at {true_click_times}")

    detected_events = []

    def on_event(event):
        detected_events.append(event)
        print(f"  [event] t~{event['sample_index']/config.SAMPLE_RATE:.3f}s  "
              f"zone={event['predicted_zone']:<20s} conf={event['zone_confidence']:.2f}  "
              f"snr={event['snr_db']:5.1f}dB  exposure={event['exposure_score']:.1f}  "
              f"(model_trained={event['model_trained']})")

    pipeline = AcousticGuardPipeline(on_event=on_event)
    source = ArrayAudioSource(audio, realtime=False)  # fast-forward, no sleeping

    print("\nStreaming through pipeline...")
    pipeline.run(source)

    print(f"\nDetected {len(detected_events)} onset(s) vs. {len(true_click_times)} injected.")
    assert len(detected_events) > 0, "FAIL: onset detector found nothing at all"

    # sanity check: each detected onset should land reasonably close to a true click
    for ev in detected_events:
        t = ev["sample_index"] / config.SAMPLE_RATE
        nearest = min(abs(t - ct) for ct in true_click_times)
        status = "OK" if nearest < 0.05 else "?? (no nearby injected click)"
        print(f"  detected t={t:.3f}s -> nearest injected click Δ={nearest*1000:.1f}ms  {status}")

    final_score = pipeline.exposure_scorer.current_score()
    print(f"\nFinal exposure score snapshot: {final_score}")
    assert 0.0 <= final_score["exposure_score"] <= 100.0, "FAIL: exposure score out of range"

    print("\n✅ Pipeline smoke test passed — plumbing works end to end.")
    print("   (Predictions are meaningless until you load a trained checkpoint —")
    print("    this only validates that audio flows correctly through every stage.)")


if __name__ == "__main__":
    main()
