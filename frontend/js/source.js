/* Event source — the one swappable seam.
 *
 * The frontend never learns which source it got: both emit the exact schema from
 * CLAUDE.md. Primary path is the real backend over ws://localhost:8765. If nothing
 * is listening, we fall back to an in-browser fake that is a faithful port of
 * exposure/risk.py + exposure/source.py, so the UI is fully demoable with zero
 * setup (double-click index.html).
 *
 * Handlers: { onEvent(keystrokeEvent), onState(stateMsg), onStatus({mode, detail}) }
 */
window.App = window.App || {};
(function (App) {
  "use strict";

  const DEFAULT_WS = "ws://localhost:8765";
  const CONNECT_TIMEOUT_MS = 1600;

  // Mirrors of the two backend constants (exposure/alert.py). Only used offline.
  // Keep these in step with alert.py — if they drift, the offline demo and the
  // live backend disagree about what counts as an alert.
  const RISK_THRESHOLD = 0.80;
  const CRITICAL_THRESHOLD = 0.95;

  const VOCAB = "abcdefghijklmnopqrstuvwxyz ".split("");
  const TOP_K = 5;
  const LOG_SIZE = 200;

  // --- faithful port of exposure/risk.py::risk_score ----------------------
  function riskScore(probs, marginWeight = 0.5) {
    if (!probs || probs.length === 0) return 0.0;
    let ranked = probs.map((p) => Math.max(0.0, p)).sort((a, b) => b - a);
    const total = ranked.reduce((a, b) => a + b, 0);
    if (total <= 0.0) return 0.0;
    ranked = ranked.map((p) => p / total);
    if (ranked.length < 2) return 0.0;

    const margin = ranked[0] - ranked[1];
    let entropy = 0.0;
    for (const p of ranked) if (p > 0.0) entropy -= p * Math.log(p);
    const normEntropy = entropy / Math.log(ranked.length);
    const peakedness = 1.0 - normEntropy;

    const score = marginWeight * margin + (1.0 - marginWeight) * peakedness;
    return Math.max(0.0, Math.min(1.0, score));
  }

  // --- faithful port of exposure/alert.py::decide_alert -------------------
  function decideAlert(risk, typingDetected) {
    if (!typingDetected) return [false, "none"];
    if (risk > CRITICAL_THRESHOLD) return [true, "critical"];
    if (risk > RISK_THRESHOLD) return [true, "moderate"];
    return [false, "none"];
  }

  // --- small RNG helpers --------------------------------------------------
  const rand = (lo, hi) => lo + Math.random() * (hi - lo);
  const randint = (lo, hi) => Math.floor(rand(lo, hi + 1));
  function sample(arr, k) {
    const pool = arr.slice();
    const out = [];
    for (let i = 0; i < k; i++) out.push(pool.splice(randint(0, pool.length - 1), 1)[0]);
    return out;
  }

  // Mirrors FakeSource._distribution: three regimes with distinct shapes.
  function distribution() {
    const r = Math.random();
    let regime;
    if (r < 0.35) regime = "clean";
    else if (r < 0.75) regime = "usable";
    else regime = "ambiguous";

    // Ranges straddle the 0.80 / 0.95 thresholds above — keep in step with
    // FakeSource._distribution in exposure/source.py.
    let leader, tailSpread;
    if (regime === "clean") { leader = rand(0.965, 0.998); tailSpread = 0.35; }
    else if (regime === "usable") { leader = rand(0.88, 0.96); tailSpread = 0.7; }
    else { leader = rand(0.22, 0.60); tailSpread = 1.0; }

    const keys = sample(VOCAB, TOP_K);
    const tailW = [];
    for (let i = 0; i < TOP_K - 1; i++) tailW.push(rand(1.0 - tailSpread, 1.0));
    const tailTotal = tailW.reduce((a, b) => a + b, 0) || 1.0;
    const remaining = 1.0 - leader;
    const probs = [leader, ...tailW.map((w) => (remaining * w) / tailTotal)];

    const pairs = keys.map((k, i) => [k, probs[i]]);
    pairs.sort((a, b) => b[1] - a[1]);
    return pairs;
  }

  // Assemble one event the same way exposure/event.py::assemble_event does.
  function assembleEvent(topk, speechPresent) {
    const probs = topk.map((kp) => kp[1]);
    const risk = riskScore(probs);
    const typing = topk.length > 0;
    const [alert, severity] = decideAlert(risk, typing);
    return {
      type: "keystroke",
      key_top1: topk[0][0],
      key_topk: topk,
      confidence: topk[0][1],
      timestamp: Date.now() / 1000,
      mode: "normal",
      risk_score: risk,
      typing_detected: typing,
      speech_present: speechPresent,
      alert: alert,
      alert_severity: severity,
    };
  }

  // --- Offline fake source: mirrors server + ExposureState so app.js code
  //     is identical whether live or offline (state msgs drive latch/stats). ---
  class FakeHub {
    constructor(handlers) {
      this.h = handlers;
      this._timer = null;
      this._burstLeft = 0;
      this._speech = false;
      this._reset();
    }
    _reset() {
      this._alerts = [];
      this._latched = false;
      this._peak = "none";
      this._events = 0;
      this._totalAlerts = 0;
    }
    _snapshot() {
      return {
        type: "state",
        latched: this._latched,
        peak_severity: this._peak,
        total_events: this._events,
        total_alerts: this._totalAlerts,
        recent_alerts: this._alerts.slice(0, LOG_SIZE),
      };
    }
    _rank(sev) { return { none: 0, moderate: 1, critical: 2 }[sev]; }
    _record(evt) {
      this._events += 1;
      if (!evt.alert) return;
      this._totalAlerts += 1;
      this._alerts.unshift(evt);
      if (this._alerts.length > LOG_SIZE) this._alerts.pop();
      this._latched = true;
      if (this._rank(evt.alert_severity) > this._rank(this._peak)) this._peak = evt.alert_severity;
    }
    start() {
      this.h.onStatus({ mode: "offline", detail: "in-browser demo" });
      this.h.onState(this._snapshot());
      this._tick();
    }
    _tick() {
      if (this._burstLeft <= 0) {
        this._burstLeft = randint(5, 14);
        this._speech = Math.random() < 0.3;
        // pause between bursts — lets the latched indicator sit lit through a gap
        this._schedule(rand(1.2, 3.0), true);
        return;
      }
      const evt = assembleEvent(distribution(), this._speech);
      this._record(evt);
      this.h.onEvent(evt);
      this._burstLeft -= 1;
      this._schedule(rand(0.06, 0.22), false);
    }
    _schedule(seconds, isPause) {
      this._timer = setTimeout(() => this._tick(), seconds * 1000);
    }
    clearLatch() {
      this._latched = false;
      this._peak = "none";
      this.h.onState(this._snapshot());
    }
    stop() { if (this._timer) clearTimeout(this._timer); }
  }

  // --- Live source: real backend over websocket ---------------------------
  class WsHub {
    constructor(url, handlers) { this.url = url; this.h = handlers; this.ws = null; }
    connect(onFail) {
      let settled = false;
      const timeout = setTimeout(() => {
        if (!settled) { settled = true; try { this.ws.close(); } catch (e) {} onFail(); }
      }, CONNECT_TIMEOUT_MS);

      try { this.ws = new WebSocket(this.url); }
      catch (e) { clearTimeout(timeout); onFail(); return; }

      this.ws.onopen = () => {
        settled = true; clearTimeout(timeout);
        this.h.onStatus({ mode: "live", detail: this.url });
      };
      this.ws.onmessage = (m) => {
        let msg; try { msg = JSON.parse(m.data); } catch (e) { return; }
        if (msg.type === "state") this.h.onState(msg);
        else if (msg.type === "keystroke") this.h.onEvent(msg);
      };
      this.ws.onerror = () => {}; // close handles fallback
      this.ws.onclose = () => {
        if (!settled) { settled = true; clearTimeout(timeout); onFail(); }
      };
    }
    clearLatch() {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "clear_latch" }));
      }
    }
  }

  // Public entry: try live, fall back to offline fake.
  App.connectSource = function (handlers) {
    const params = new URLSearchParams(location.search);
    const url = params.get("ws") || DEFAULT_WS;
    const forceOffline = params.get("offline") === "1";

    let active;
    const startFake = () => { active = new FakeHub(handlers); active.start(); };

    if (forceOffline) { startFake(); }
    else {
      handlers.onStatus({ mode: "connecting", detail: url });
      const live = new WsHub(url, handlers);
      active = live;
      live.connect(startFake);
    }
    // Controller the app uses regardless of which source won.
    return { clearLatch: () => active && active.clearLatch && active.clearLatch() };
  };

  // Exported for reuse/testing parity.
  App.risk = { riskScore, decideAlert, RISK_THRESHOLD, CRITICAL_THRESHOLD };
})(window.App);
