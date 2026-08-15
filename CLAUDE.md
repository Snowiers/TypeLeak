# CLAUDE.md

## Read this first

This is a hackathon project with roughly 32 hours remaining, including sleep. The team is
working in parallel against fixed interface contracts, split across whichever people end up
owning which piece.

**Do not write implementation code until explicitly asked.** When we start a task, first
confirm you understand the relevant contract below, ask about anything ambiguous, then
propose an approach. Wait for confirmation before writing files.

Bias toward **small, working, testable pieces** over complete-but-untested systems. A
degraded feature that runs beats a complete feature that doesn't.

Work is not yet split among the team. This file describes the system, not who owns which
part.

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
content.** Knowing "this typing is exploitable" does not require actually reading the password
— the alert should fire on the *existence and strength* of an exploitable acoustic signal, not
on a successful decode. The alert trigger is built from two things, both independent of
transcription content:

1. Is there acoustically identifiable typing signal at all, and how strong/clean is it (a risk
   score, not a transcript)
2. Is a sensitive input field currently focused (the context signal, from an OS-level hook,
   never derived from audio content)

These two signals feed the alert decision. The attack twin's transcription output is a
*separate* branch off the same raw audio, shown to the user as the proof panel — it is never
an input to whether an alert fires. Do not let this separation blur during implementation.

---

## Interface contracts

These are frozen. Changing one requires agreement from everyone it affects. Code against the
contract, not against another person's internals.

### Context signal (cross-platform, one interface, per-OS backends)

There is no single mechanism that detects "a sensitive field is focused" identically across
macOS, Windows, and Linux — the underlying OS APIs are genuinely different (Accessibility API
on macOS, UI Automation on Windows, AT-SPI/X11 on Linux, and Wayland in particular is often
restrictive about this kind of introspection). The fix is one common interface with backends
selected per platform, plus a manual fallback so the demo never breaks on a platform quirk.

```
get_context()  →
  {
    "sensitive_field": bool,
    "app_name": str,
    "confidence": float,
    "source": "accessibility" | "uia" | "atspi" | "manual_fallback"
  }
```

Only build a real backend for whichever OS you are actually demoing on. Stub the others to
return `source: "manual_fallback"`. The manual fallback is a hotkey or on-screen toggle a team
member presses during the demo to simulate "sensitive field focused" — this must exist and
work regardless of whether the real OS hook is finished, because a live demo cannot depend on
an OS accessibility permission dialog behaving correctly on stage.

### Audio event (backend classifier → frontend, single schema serves both panels)

Emitted over a local websocket, one event per detected keystroke:

```
{
  "type":         "keystroke",
  "key_top1":     str,
  "key_topk":     [[key, prob], ...],
  "confidence":   float,
  "timestamp":    float,
  "mode":         "normal" | "password",
  "risk_score":   float,
  "alert":        bool,
  "alert_severity": "none" | "moderate" | "critical"
}
```

`mode` reflects the context signal at the time of the event and determines which correction
strategy the transcript panel should be displaying (see Language model correction below).
`alert` and `alert_severity` describe whether this event crossed the alert threshold and how
seriously. There is no `action` field and no audio modification anywhere in this schema — the
system observes and reports, it does not act on the audio stream.

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

### Audio conditioning (shared by both the transcript and alert branches)

Raw mic buffer in, cleaned keystroke-relevant segments out. This does not need to be a
learned separation model — keystrokes are impulsive, broadband, short transients; speech is
continuous and harmonic; most background noise is spectrally stationary. That difference is
exploitable with standard DSP:

1. **Voice activity detection** flags speech-present frames — a fast, existing VAD component
   (not custom-trained), used both to help isolate keystroke energy and as context for the
   alert severity (see below).
2. **Spectral subtraction / noise gating** against an estimated noise floor removes
   stationary background noise.
3. **Onset detection** (spectral flux or high-frequency content) finds keystroke-like
   transients in what remains.

A learned source-separation model is a legitimate stretch goal, not a requirement. Do not
attempt it before the DSP pipeline above is working end to end.

### Classification and correction

Per-keystroke log-mel spectrogram, plus inter-keystroke timing as an auxiliary feature, into
a classifier. Fine-tune rather than hand-roll where possible — using a real NVIDIA-ecosystem
component here (rather than a bespoke CNN) matters for how this project is scored, not just
how it performs.

**Correction is mode-dependent, and this is a deliberate design choice, not an afterthought:**

- `mode: "normal"` — full dictionary/language-model correction. Passwords are not English
  words; general typed text is. This is where the readable transcript comes from.
- `mode: "password"` — **no dictionary correction.** Running word-level correction on a
  password actively produces a wrong but plausible-looking result, which is worse than no
  correction at all. Use, at most, a keyboard-adjacency prior (some key transitions are
  physically faster/more common than others, independent of language) — not a word dictionary.

The transcript panel should visibly show which mode is active. Do not silently apply the same
correction strategy regardless of context — that is the single most avoidable accuracy and
credibility mistake this project could make.

### Risk and alert decision

Combines the classifier's confidence/entropy on the current audio with the context signal to
produce a risk score, then an alert severity. Whether the risk score is a genuinely separate
small model or computed directly from classifier output statistics is an open decision — see
Open decisions below.

**Alert severity table** (speech presence comes from the VAD in audio conditioning, not from
whether words were successfully transcribed):

| Speech present | Typing detected | Risk above threshold | Sensitive field | Alert |
|---|---|---|---|---|
| — | No | — | — | none |
| — | Yes | No | — | none, log only |
| Yes or No | Yes | Yes | No | moderate |
| Yes or No | Yes | Yes | Yes | critical |

Speech presence does not suppress the alert the way it would have suppressed a mitigation
action — the point of this system is to tell the truth about exposure regardless of what else
is happening in the audio. If typing is acoustically exploitable while someone is also
talking, that is still worth surfacing, arguably more so.

One sensitivity setting, not several: **risk threshold** — how confident the exposure signal
needs to be before an alert fires at all. Keep this simple; there is no masking aggressiveness
knob in this version because there is no masking.

### Alerting

This is the product. Every alert-worthy event should be visible immediately, with enough
detail to be self-explanatory: what was detected, how confident, whether a sensitive field was
focused, and when. A running log of recent alerts, plus a live "currently exposed" indicator
during active typing, is the core deliverable — treat it with the same priority as the
classifier itself, not as an add-on.

---

## Frontend

One websocket event stream (see Audio event above), three panels. Build in this order — each
is independently useful, so stopping partway still leaves something demoable:

1. **Transcript panel.** Raw per-key stream and mode-dependent corrected text, side by side,
   with a visible label showing which correction mode is active. This is the proof that
   detection works — build this first.
2. **Alert panel.** Live "currently exposed" indicator plus a scrolling log of past alerts
   with severity and reasoning. This is the actual deliverable — it directly answers "why
   does this matter" for anyone watching the demo.
3. **3D keyboard visualization.** Stylized, not realistic — simple box geometry per key in a
   grid, using three.js. On each event, flash the predicted key with opacity scaled to
   confidence. This is demo polish, not something any judging criterion strictly requires.
   If time runs short, degrade to a flat 2D HTML/CSS key grid that does the same job with a
   fraction of the build effort — this degradation path should be assumed from the start, not
   discovered under time pressure.

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

---

## Constraints and conventions

**Consent and scope, restated because it governs demo design.** All attack-twin
demonstration happens on your own team's typing, on your own hardware, live, with consent.
Never on a stranger's audio, never on anyone's pre-recorded typing without their agreement.

**No audio modification, anywhere.** This build observes and alerts only. Do not add muting,
ducking, masking, or any other change to the audio stream, even as a "quick" addition —
that is a different project and is explicitly out of scope here (see below).

**Demo reliability over cleverness.** The manual context-signal fallback must work reliably
before any stretch goal is attempted. A judge watching a live demo will remember a broken OS
hook far more than they'll credit an ambitious feature that almost worked.

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

- **Audio/backend stack** (Python audio libraries, NeMo integration specifics): owned by a
  specific team member, not yet finalized. Confirm with them before writing this stage.
- **Risk score source**: whether the exposure/risk score is a genuinely separate small model,
  or computed directly from the attack-twin classifier's confidence/entropy on the current
  audio. Either is legitimate; pick one deliberately and document the choice once made.
- **Which OS is the real context-signal backend built for**: depends on what the demo machine
  actually runs. Confirm before starting that stage.

---

## Decision gates

Relative to project start, not fixed clock times — adjust to when you actually begin, and
build in sleep. Hitting these on time matters more than any individual feature.

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
- Adversarial audio perturbation against the classifier's feature extraction.
- Dictionary/word-level correction applied to password-mode transcription, ever.
- Testing the attack twin on anyone besides consenting team members on their own hardware.
- Real OS backends for platforms you are not actually demoing on — stub them.
- Multiple keyboards or typists in the training data.
- Any feature added after the feature-freeze gate.
