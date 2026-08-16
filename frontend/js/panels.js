/* The data panels.
 *
 *  Transcript — the live "what's being typed" line. Click Finish to capture the
 *  current text into the persistent session log (kept in localStorage so logs
 *  survive reloads). Correction is off per CLAUDE.md, so this is the raw stream.
 *
 *  Exposure — the glass panel on the right. Shows the latched state
 *  (secure / exposed / critical), a short "why", and a live meter of the latest
 *  keystroke's exposure. confidence and risk_score stay distinct — risk drives this.
 */
window.App = window.App || {};
(function (App) {
  "use strict";

  const LOG_KEY = "exposure_logs";

  function loadLogs() {
    try { return JSON.parse(localStorage.getItem(LOG_KEY)) || []; }
    catch (e) { return []; }
  }
  function saveLogs(logs) {
    try { localStorage.setItem(LOG_KEY, JSON.stringify(logs)); } catch (e) {}
  }
  function fmtTime(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return `${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  // ---- Transcript + session history -------------------------------------
  class Transcript {
    constructor() {
      this.typed = document.getElementById("typed");
      this.list = document.getElementById("history-list");
      this.buffer = "";
      this.logs = loadLogs();  // [{ t, text }], newest first
      this._renderHistory();
      this._renderTyped();
    }

    push(evt) {
      this.buffer += evt.key_top1;
      this._renderTyped();
    }

    _renderTyped() {
      this.typed.textContent = this.buffer;
      this.typed.scrollTop = this.typed.scrollHeight;
    }

    finish() {
      const text = this.buffer.trim();
      if (!text) return;
      this.logs.unshift({ t: Date.now() / 1000, text });
      if (this.logs.length > 100) this.logs.pop();
      saveLogs(this.logs);
      this.buffer = "";
      this._renderTyped();
      this._renderHistory();
    }

    clearLogs() {
      this.logs = [];
      saveLogs(this.logs);
      this._renderHistory();
    }

    _renderHistory() {
      this.list.innerHTML = "";
      if (this.logs.length === 0) {
        const li = document.createElement("li");
        li.className = "hist-empty";
        li.textContent = "nothing captured yet";
        this.list.appendChild(li);
        return;
      }
      for (const entry of this.logs) {
        const li = document.createElement("li");
        li.className = "hist-item";
        const time = document.createElement("span");
        time.className = "hist-time";
        time.textContent = fmtTime(entry.t);
        const text = document.createElement("span");
        text.className = "hist-text";
        text.textContent = entry.text;
        li.appendChild(time);
        li.appendChild(text);
        this.list.appendChild(li);
      }
    }
  }

  // ---- Exposure overlay --------------------------------------------------
  const SEV_RANK = { none: 0, moderate: 1, critical: 2 };
  const STATE_WORD = { secure: "secure", moderate: "exposed", critical: "critical" };
  // short, human "why" — not too detailed
  const WHY = {
    secure: "typing sound is too ambiguous to reconstruct",
    moderate: "individual keystrokes are acoustically identifiable",
    critical: "typing is cleanly reconstructable from the audio",
  };
  const RISK_T = (App.risk && App.risk.RISK_THRESHOLD) || 0.55;
  const CRIT_T = (App.risk && App.risk.CRITICAL_THRESHOLD) || 0.80;

  const LOG_MAX = 40;
  const LOG_INTERVAL_MS = 10000;   // at most one alert entry every ~10s
  const PHRASE_MAX = 48;           // characters of reconstructed phrase to show

  function fmtClock(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }
  // per CLAUDE.md: a false VAD flag may just mean the model didn't surface it —
  // label it "unknown" rather than claiming confirmed silence.
  function speechLabel(on) {
    return on ? "speech present" : "speech: unknown";
  }

  class Exposure {
    constructor() {
      this.panel = document.getElementById("exposure");
      this.stateEl = document.getElementById("exp-state");
      this.whyEl = document.getElementById("exp-why");
      this.meter = document.getElementById("exp-meter");
      this.logEl = document.getElementById("exp-log");

      this.latched = false;   // stays lit until cleared
      this.peak = "none";     // worst severity since the last clear
      this.lastRisk = 0;      // most recent keystroke's risk (drives the meter)
      this.recent = [];       // logged alerts, newest first — one per ~10s window

      // accumulates the current window (reconstructed phrase + its worst stats)
      this.buf = "";
      this.winRisk = 0;
      this.winSev = "none";
      this.winSpeech = false;
      this.lastLogMs = 0;     // 0 → first alert logs right away

      this._render();
      this._renderLog();
    }

    // Authoritative state (on connect / after clear). Only resyncs the latched
    // indicator; the phrase log is a frontend running record and survives clears.
    applyState(state) {
      this.latched = !!state.latched;
      this.peak = state.peak_severity || "none";
      this._render();
    }

    applyEvent(evt) {
      this.lastRisk = evt.risk_score;

      // build up the reconstructed phrase + worst stats for this window
      this.buf += evt.key_top1;
      if (evt.risk_score > this.winRisk) this.winRisk = evt.risk_score;
      if (evt.speech_present) this.winSpeech = true;

      if (evt.alert) {
        this.latched = true;
        if (SEV_RANK[evt.alert_severity] > SEV_RANK[this.peak]) this.peak = evt.alert_severity;
        if (SEV_RANK[evt.alert_severity] > SEV_RANK[this.winSev]) this.winSev = evt.alert_severity;

        // only log an alert (with the exploitable phrase) every ~10 seconds
        const now = Date.now();
        if (now - this.lastLogMs >= LOG_INTERVAL_MS) {
          this._logPhrase(evt.timestamp);
          this.lastLogMs = now;
        }
      }
      this._render();
    }

    // Emit one alert entry for the window: the phrase that could be reconstructed,
    // its worst severity/risk, whether speech was present, and when. Then reset.
    _logPhrase(ts) {
      const phrase = this.buf.replace(/\s+/g, " ").trim().slice(-PHRASE_MAX);
      this.recent.unshift({
        phrase: phrase || "…",
        sev: this.winSev === "none" ? "moderate" : this.winSev,
        risk: this.winRisk,
        speech: this.winSpeech,
        t: ts,
      });
      if (this.recent.length > LOG_MAX) this.recent.pop();
      this.buf = "";
      this.winRisk = 0;
      this.winSev = "none";
      this.winSpeech = false;
      this._renderLog();
    }

    // The panel word reflects the latched worst severity.
    _severityClass() {
      if (this.latched && this.peak === "critical") return "critical";
      if (this.latched && this.peak === "moderate") return "moderate";
      return "secure";
    }

    _render() {
      const cls = this._severityClass();
      this.panel.classList.remove("secure", "moderate", "critical");
      this.panel.classList.add(cls);
      this.stateEl.textContent = STATE_WORD[cls];
      this.whyEl.textContent = WHY[cls];

      // meter reflects the latest keystroke's exposure, colored by its own band
      const pct = Math.max(0, Math.min(1, this.lastRisk)) * 100;
      this.meter.style.width = pct.toFixed(0) + "%";
      this.meter.classList.remove("mid", "hi");
      if (this.lastRisk > CRIT_T) this.meter.classList.add("hi");
      else if (this.lastRisk > RISK_T) this.meter.classList.add("mid");

      document.body.classList.toggle("alarm-critical", cls === "critical");
      if (App._onMood) App._onMood(cls);
    }

    // Running log — each entry: severity, time, the reconstructable phrase, and
    // the risk + speech context for that window.
    _renderLog() {
      if (!this.logEl) return;
      this.logEl.innerHTML = "";
      if (this.recent.length === 0) {
        const li = document.createElement("li");
        li.className = "al-empty";
        li.textContent = "no alerts yet";
        this.logEl.appendChild(li);
        return;
      }
      for (const a of this.recent) {
        const li = document.createElement("li");
        li.className = "al-item " + a.sev;

        const head = document.createElement("div");
        head.className = "al-head";
        const sev = document.createElement("span");
        sev.className = "al-sev";
        sev.textContent = a.sev;
        const time = document.createElement("span");
        time.className = "al-time";
        time.textContent = fmtClock(a.t);
        head.appendChild(sev);
        head.appendChild(time);

        const phrase = document.createElement("div");
        phrase.className = "al-phrase";
        phrase.textContent = `"${a.phrase}"`;

        const meta = document.createElement("div");
        meta.className = "al-meta";
        meta.textContent = `risk ${a.risk.toFixed(2)} · ${speechLabel(a.speech)}`;

        li.appendChild(head);
        li.appendChild(phrase);
        li.appendChild(meta);
        this.logEl.appendChild(li);
      }
    }
  }

  App.Transcript = Transcript;
  App.Exposure = Exposure;
})(window.App);
