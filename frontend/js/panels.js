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

  class Exposure {
    constructor() {
      this.panel = document.getElementById("exposure");
      this.stateEl = document.getElementById("exp-state");
      this.whyEl = document.getElementById("exp-why");
      this.meter = document.getElementById("exp-meter");

      this.latched = false;   // stays lit until cleared
      this.peak = "none";     // worst severity since the last clear
      this.lastRisk = 0;      // most recent keystroke's risk (drives the meter)
      this._render();
    }

    // Authoritative state (on connect / after clear). Resync.
    applyState(state) {
      this.latched = !!state.latched;
      this.peak = state.peak_severity || "none";
      this._render();
    }

    applyEvent(evt) {
      this.lastRisk = evt.risk_score;
      if (evt.alert) {
        this.latched = true;
        if (SEV_RANK[evt.alert_severity] > SEV_RANK[this.peak]) this.peak = evt.alert_severity;
      }
      this._render();
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
  }

  App.Transcript = Transcript;
  App.Exposure = Exposure;
})(window.App);
