# CLAUDE.md

## Read this first

This is a hackathon project with roughly 32 hours remaining, including sleep. Started
2026-08-15. The team is working in parallel against fixed interface contracts, split across
whichever people end up owning which piece.

**Do not write implementation code until explicitly asked.** When we start a task, first
confirm you understand the relevant contract below, ask about anything ambiguous, then
propose an approach. Wait for confirmation before writing files.

Bias toward **small, working, testable pieces** over complete-but-untested systems. A
degraded feature that runs beats a complete feature that doesn't.

Work is not yet split among the team. This file describes the system, not who owns which
part. One known boundary: **audio capture, conditioning, and classification are owned by a
teammate on the model side.** Everything downstream of the emitted audio event — risk, alert
decision, transport, frontend — is on this side of the seam. The seam may still move; keep
the coupling to the event schema, not to anyone's internals.

---

## What we are building

An **acoustic side-channel exposure monitor**: a real-time system that detects when someone's
typing can be reconstructed from ambient microphone audio (a published, real attack — keyboard
acoustic emanations can be classified per-key with high accuracy) and surfaces an alert showing
that it is happening, live.

**This system does not intervene.** It does not mute, mask, duck, or otherwise modify the
audio stream in any way. Its entire job is detection and disclosure: prove, in real time, that
an ordinary microphone is leaking enough information to reconstruct typed text, and show the
person exactly when and how exposed they are. Any active mitigation is future work, not part
of this build — do not design around muting, masking, or otherwise touching the outgoing
audio anywhere in this system.

Two halves, sharing one audio pipeline:

**The attack twin.** A classifier that reconstructs keystrokes from acoustic signal. This is
not a side experiment — it is the core proof that the threat is real, and its output (the
reconstructed transcript) is one of the two primary things this project shows a user. It is
trained and tested only on our own team's typing, on our own hardware, with consent, live at
the booth. Never on a stranger, never on pre-recorded audio of someone who didn't agree to it.
This constraint is not a footnote — it governs every design decision about how the attack side
is demoed.

**The exposure monitor.** A real-time pipeline that listens to the same audio a call or
recording app would capture, continuously estimates how acoustically exploitable the current
typing is, and raises a visible alert when exposure crosses a threshold — nothing more. The
alert is the product.

### Why this framing, not "we built a keylogger"

Most published work in this space is attack-only, built to prove reconstruction is possible.
Almost none of it is framed as something a user would actually see running, showing them their
own exposure as it happens. That is the actual gap this project fills: not a countermeasure,
but a **visibility tool** — the acoustic equivalent of a webcam light. It doesn't have to stop
the leak to be valuable; making an invisible leak visible is itself the useful thing, the same
way a webcam indicator light doesn't stop a camera from working but makes covert use much
harder to get away with.

### The architectural rule that makes this trustworthy

**The alert decision must never depend on the attack twin having successfully transcribed the
content.** Knowing "this typing is exploitable" does not require actually reading what was
typed — the alert fires on the *existence and strength* of an exploitable acoustic signal, not
on a successful decode.

This version has no context signal (see Explicitly out of scope), so the alert rests on one
input: **a risk score describing whether there is acoustically identifiable typing signal at
all, and how strong and clean it is.** A risk score, never a transcript.

The line between the risk score and the transcript is about *content versus statistics*. The
risk score may consume the shape of the classifier's output distribution — top-1 margin,
entropy, onset strength. It must never consume the predicted key identity or any decoded text.
The attack twin's transcription output is a *separate* branch off the same raw audio, shown to
the user as the proof panel — it is never an input to whether an alert fires. Do not let this
separation blur during implementation.

---

## Interface contracts

These are frozen. Changing one requires agreement from everyone it affects. Code against the
contract, not against another person's internals.

### Audio event (backend classifier → frontend, single schema serves both panels)

Emitted over a local websocket, one event per detected keystroke:

```
{
  "type":            "keystroke",
  "key_top1":        str,
  "key_topk":        [[key, prob], ...],
  "confidence":      float,     // 0-1, classifier top-1 probability == key_topk[0][1]
  "timestamp":       float,
  "mode":            "normal",  // always "normal" in this version, see below
  "risk_score":      float,     // 0-1 exposure estimate, distribution shape only
  "typing_detected": bool,
  "speech_present":  bool,      // from VAD; false if the model side does not surface it
  "alert":           bool,
  "alert_severity":  "none" | "moderate" | "critical"
}
```

`confidence` is the classifier's own top-1 probability. `risk_score` is the independent
exposure estimate — how exploitable this acoustic signal is — and is what the alert decision
consumes. They are different numbers and must not be conflated in code or in the UI.

`mode` is retained in the schema but is always `"normal"` in this version, since there is no
context signal and no password mode. Do not delete the field; a future version needs it.

`typing_detected` and `speech_present` are explicit rather than inferred from `alert_severity`,
because the alert panel has to show *why* an alert fired and severity alone is lossy. If the
model side does not surface VAD output, `speech_present` is `false` and the panel labels it
unknown rather than silently claiming silence.

There is no `action` field and no audio modification anywhere in this schema — the system
observes and reports, it does not act on the audio stream.

### Training data format

```
labeled_dataset/
  <sample_id>.wav
  labels.csv:  sample_id, key_label, onset_timestamp
```

One keyboard, one typist, for this hackathon. Do not scope this up to multiple keyboards or
typists — the classifier only needs to prove the concept, and expanding the data collection
target is the single easiest way to run out of time before anything works end to end.

---

## Pipeline mechanics

### Audio conditioning (owned by the model side)

Raw mic buffer in, cleaned keystroke-relevant segments out. This does not need to be a
learned separation model — keystrokes are impulsive, broadband, short transients; speech is
continuous and harmonic; most background noise is spectrally stationary. That difference is
exploitable with standard DSP:

1. **Voice activity detection** flags speech-present frames — a fast, existing VAD component
   (not custom-trained), used both to help isolate keystroke energy and to populate
   `speech_present` on the event.
2. **Spectral subtraction / noise gating** against an estimated noise floor removes
   stationary background noise.
3. **Onset detection** (spectral flux or high-frequency content) finds keystroke-like
   transients in what remains.

A learned source-separation model is a legitimate stretch goal, not a requirement. Do not
attempt it before the DSP pipeline above is working end to end.

### Classification

Per-keystroke log-mel spectrogram, plus inter-keystroke timing as an auxiliary feature, into
a classifier. Fine-tune rather than hand-roll where possible — using a real NVIDIA-ecosystem
component here (rather than a bespoke CNN) matters for how this project is scored, not just
how it performs.

**No correction of any kind in this version.** No dictionary, no language model, no
keyboard-adjacency prior. The transcript panel shows the raw per-key stream, with the
correction column present but passthrough and visibly labeled as such. Rationale: correction
was only ever justified as mode-dependent, and without a context signal there is no mode to
depend on. Reserving the column keeps re-enabling it a one-place change rather than a layout
rebuild.

**If correction is ever turned back on, it comes back mode-aware, not global.** Word-level
correction applied to a password produces a wrong but plausible-looking result, which is worse
than no correction at all. That constraint outlives this version.

### Risk and alert decision

The risk score is **computed directly from the attack-twin classifier's output statistics** —
top-1 margin over top-2, plus normalized entropy — not from a separate model. Decision made
deliberately: a separate model needs its own labels, which do not exist and cannot be
collected before the +20h gate. Documented here as the resolution of a previously open
decision.

The risk score consumes distribution shape only. It never reads `key_top1`, `key_topk` key
identities, or any decoded text — see the architectural rule above.

**Alert severity table.** Severity is bound to two thresholds on the risk score. Speech
presence comes from the VAD, not from whether words were successfully transcribed, and does
not suppress the alert.

| Typing detected | Risk score | Speech present | Alert |
|---|---|---|---|
| No | — | — | none |
| Yes | below `risk_threshold` | — | none, log only |
| Yes | above `risk_threshold` | Yes or No | moderate |
| Yes | above `critical_threshold` | Yes or No | critical |

Speech presence does not suppress the alert the way it would have suppressed a mitigation
action — the point of this system is to tell the truth about exposure regardless of what else
is happening in the audio. If typing is acoustically exploitable while someone is also
talking, that is still worth surfacing, arguably more so.

Two constants, both on the same axis: **`risk_threshold`** (how confident the exposure signal
must be before an alert fires at all) and **`critical_threshold`** (above which exposure is
severe). The severity axis is "how exploitable," purely acoustic. There is no masking
aggressiveness knob in this version because there is no masking.

### Alerting

This is the product. Every alert-worthy event should be visible immediately, with enough
detail to be self-explanatory: what was detected, how confident, the risk score, whether
speech was present, and when. A running log of recent alerts, plus a live "currently exposed"
indicator during active typing, is the core deliverable — treat it with the same priority as
the classifier itself, not as an add-on.

The "currently exposed" indicator **latches**. No event is emitted when typing stops, so the
indicator stays lit until manually cleared with `Ctrl+Shift+X`, active only while the frontend
window has focus. No global OS hotkey — nothing to break on stage.

---

## Frontend

Vanilla HTML/CSS/JS, no framework, no build step. Websocket client to `ws://localhost:8765`,
configurable. A build toolchain is pure risk at hour zero; this is reversible if it becomes
limiting.

The event source is swappable behind one interface: a fake generator now, the real websocket
later. Nothing in the panels couples to anything but the event schema above.

One event stream, three panels. Build in this order — each is independently useful, so
stopping partway still leaves something demoable:

1. **Transcript panel.** Two columns: raw per-key stream on the left, corrected text on the
   right. Correction is off, so the right column is passthrough and labeled as such, with the
   mode indicator showing `normal`. This is the proof that detection works — build this first.
2. **Alert panel.** Live latched "currently exposed" indicator plus a scrolling log of past
   alerts with severity and reasoning. This is the actual deliverable — it directly answers
   "why does this matter" for anyone watching the demo.
3. **3D keyboard visualization.** Stylized, not realistic — simple box geometry per key in a
   grid, using three.js loaded as a local file. On each event, flash the predicted key with
   opacity scaled to confidence. This is demo polish, not something any judging criterion
   strictly requires. If time runs short, degrade to a flat 2D HTML/CSS key grid that does the
   same job with a fraction of the build effort — this degradation path should be assumed from
   the start, not discovered under time pressure.

All three panels can and should be built against hand-written fake events before the backend
pipeline is finished. Do not let frontend work block on backend completion.

---

## Validation

There is no attack-vs-defense closed loop in this version, since there is no defense. Instead,
validate that the **risk score means something**: show that it correlates with actual
reconstruction accuracy — i.e., when the system reports high exposure risk, the attack twin's
transcription really is accurate, and when it reports low risk, the transcription really is
poor or absent. This is what proves the alert isn't just a generic "typing detected" flag but
a genuine estimate of how exploitable that typing was.

Report it as a concrete comparison: risk score vs. measured transcription accuracy across a
held-out set of samples.

**This requires deliberately collecting low-risk samples** — noisy audio, distant mic, speech
over typing — not only clean ones. A held-out set of uniformly clean samples produces a single
cluster on the plot and proves nothing. Plan the bad audio during data collection, not after.

---

## Constraints and conventions

**Consent and scope, restated because it governs demo design.** All attack-twin
demonstration happens on your own team's typing, on your own hardware, live, with consent.
Never on a stranger's audio, never on anyone's pre-recorded typing without their agreement.

**No audio modification, anywhere.** This build observes and alerts only. Do not add muting,
ducking, masking, or any other change to the audio stream, even as a "quick" addition —
that is a different project and is explicitly out of scope here (see below).

**No OS-specific code.** With the context signal cut, nothing in this system touches platform
APIs. It is cross-platform by virtue of not needing to be.

**Demo reliability over cleverness.** A judge watching a live demo will remember a broken
component far more than they'll credit an ambitious feature that almost worked. Every stage
must have a working degraded path before any stretch goal is attempted.

**NVIDIA ecosystem.** Prefer a real NeMo audio model for the classifier over a hand-rolled
CNN — this is both a technical-quality choice and a scoring one. Local, on-device inference
is the actual reason this project needs the Spark: real-time audio interception has a hard
round-trip latency budget (capture → condition → classify → decide → alert) that has to
complete before the next audio frame, with no cloud round-trip. This is the Spark story —
lead with latency, not "everything fits in memory."

**Alerts are not optional polish.** They are the deliverable. Build the alert panel early and
treat it as at least as important as classifier accuracy.

---

## Open decisions — do not assume, ask

- **Team split.** Who owns what beyond "the model side is a teammate's" is undecided. The seam
  may move. Keep coupling to the event schema, not to internals.
- **Audio/backend stack** (Python audio libraries, NeMo integration specifics, capture library,
  sample rate, buffer size): owned by the model-side teammate, not yet finalized. The latency
  budget in milliseconds falls out of the buffer size and is still unknown.
- **VAD component** (Silero vs. WebRTC): model side's call.
- **Whether the model side surfaces `speech_present`** on the event. If not, it is `false` and
  the alert panel labels it unknown.

Resolved since the first draft: risk score is derived from classifier statistics, not a
separate model. Websocket at `ws://localhost:8765`, vanilla frontend, no build step.

---

## Decision gates

Relative to project start (2026-08-15). Adjust to when you actually begin, and build in sleep.
Hitting these on time matters more than any individual feature.

| Offset | Gate | If missed |
|---|---|---|
| +2h | Contracts confirmed by everyone, fake-event frontend and synthetic-audio backend testing started in parallel | Stop everything else and finish this first |
| +8h | Attack twin reconstructs keystrokes from clean audio (no noise/speech) at all | This is the minimum viable core — do not move on without it |
| +14h | Audio conditioning (VAD + onset detection) integrated; classifier works on real, noisier audio | If not, ship the clean-audio version as the demo baseline |
| +20h | Risk/alert decision engine wired end to end, alerts firing and visible in the frontend | This is the entire deliverable — protect this above all stretch goals |
| +26h | Validation complete: risk score shown correlating with actual reconstruction accuracy | This is your strongest piece of evidence — do not skip it for frontend polish |
| +29h | Feature freeze | No exceptions |
| +31h | Full demo rehearsal complete | Cut features until it runs clean |

---

## Explicitly out of scope

- Any active mitigation: muting, ducking, masking, or otherwise modifying the outgoing audio.
  This was part of an earlier version of the project and has been deliberately cut. If it
  comes back later, it is a new, separate feature, not a quick addition to this build.
- **The context signal in its entirety.** No sensitive-field detection, no `get_context()`, no
  per-OS accessibility backends (Accessibility API / UI Automation / AT-SPI), no manual
  fallback toggle. Cut deliberately. The alert decision is acoustic-only.
- **Password mode.** `mode` is always `"normal"`. No password detection of any kind.
- **All transcript correction.** No dictionary, no language model, no keyboard-adjacency prior.
  If it returns, it returns mode-aware — dictionary/word-level correction applied to
  password-mode transcription, ever, stays permanently out of scope.
- Adversarial audio perturbation against the classifier's feature extraction.
- Testing the attack twin on anyone besides consenting team members on their own hardware.
- Multiple keyboards or typists in the training data.
- Any feature added after the feature-freeze gate.
