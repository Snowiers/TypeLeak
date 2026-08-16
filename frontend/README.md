# frontend — acoustic exposure monitor UI

Vanilla HTML/CSS/JS. No framework, no build step (per CLAUDE.md). Minimal, rounded,
near-black with a faint starfield. A Mac keyboard is the centerpiece — keys flash on
the predicted keystroke — with a live "what's being typed" line above it and a small
square exposure overlay in the bottom-right.

Observe-only. Nothing here modifies audio; it visualizes detection and disclosure.

## Run it

**Offline (zero setup):** open `index.html` in a browser. With no backend listening it
falls back to an in-browser fake source — a faithful port of `exposure/risk.py` and the
`exposure/source.py` regimes (verified to produce identical risk scores).

**Live (real backend):**
```
python -m exposure                 # serves events on ws://localhost:8765
# then open frontend/index.html
```
The source pill top-right shows `live` vs `demo`.

Serving over http (avoids file:// quirks):
```
cd frontend && python3 -m http.server 8000   # then open http://localhost:8000
```

### Query params
- `?ws=ws://host:port` — point at a non-default backend.
- `?offline=1` — force the in-browser fake even if a backend is up.

## The pieces
- **Typed line + history.** Live raw per-key stream (correction is off per CLAUDE.md).
  Click **finish** to capture the current text into the session log on the left; logs
  persist across reloads via `localStorage`. `clear` empties them.
- **Mac keyboard.** Full US layout (function row, shifted number legends, modifiers,
  arrows). Only letters + space light up — that's the model's VOCAB. Flash brightness
  scales with **confidence**; tint follows severity.
- **Exposure overlay** (bottom-right). Latched `secure → exposed → critical` state and
  an alert count; click it to see recent alerts (risk score + time). This is the
  deliverable — the "currently exposed" indicator that latches.

## Wiring notes
- Couples only to the event schema; source swappable behind `App.connectSource`.
- **`confidence` drives key brightness; `risk_score` drives the alarm** — never conflated.
- `speech_present: false` from a source without VAD is treated as unknown, not silence.
- The alarm **latches** — lit until acknowledged with `Ctrl+Shift+X` (also `Cmd+Shift+X`
  on macOS), window-focus only, no global hotkey. Sends `{"type":"clear_latch"}`.
