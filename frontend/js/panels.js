/* The data panels.
 *
 *  Transcript — the live "what's being typed" line. Click Finish to capture the
 *  current text into the persistent session log (kept in localStorage so logs
 *  survive reloads). Correction is off per CLAUDE.md, so this is the raw stream.
 *
 *  Exposure — the small square overlay, bottom-right. Shows the latched state
 *  (secure / exposed / critical) and an alert count; click it to see recent alerts.
 *  confidence and risk_score stay visually distinct — risk drives the alarm.
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

  class Exposure {
    constructor() {
      this.box = document.getElementById("alert");
      this.stateEl = document.getElementById("alert-state");
      this.countEl = document.getElementById("alert-count");
      this.list = document.getElementById("alert-list");

      this.latched = false;
      this.peak = "none";
      this.alerts = 0;
      this.recent = [];

      this.box.addEventListener("click", () => this.box.classList.toggle("open"));
      this._render();
    }

    // Authoritative state (on connect / after clear). Resync.
    applyState(state) {
      this.latched = !!state.latched;
      this.peak = state.peak_severity || "none";
      this.alerts = state.total_alerts || 0;
      this.recent = (state.recent_alerts || []).slice(0, 40);
      this._render();
      this._renderList();
    }

    applyEvent(evt) {
      if (evt.alert) {
        this.alerts += 1;
        this.latched = true;
        if (SEV_RANK[evt.alert_severity] > SEV_RANK[this.peak]) this.peak = evt.alert_severity;
        this.recent.unshift(evt);
        if (this.recent.length > 40) this.recent.pop();
        this._renderList();
      }
      this._render();
    }

    _severityClass() {
      if (this.latched && this.peak === "critical") return "critical";
      if (this.latched && this.peak === "moderate") return "moderate";
      return "secure";
    }

    _render() {
      const cls = this._severityClass();
      this.box.classList.remove("secure", "moderate", "critical");
      this.box.classList.add(cls);
      this.stateEl.textContent = STATE_WORD[cls];
      this.countEl.textContent = this.alerts === 0 ? "watching" :
        this.alerts + (this.alerts === 1 ? " alert" : " alerts");
      document.body.classList.toggle("alarm-critical", cls === "critical");
      if (App._onMood) App._onMood(cls);
    }

    _renderList() {
      this.list.innerHTML = "";
      if (this.recent.length === 0) {
        const li = document.createElement("li");
        li.className = "al-empty";
        li.textContent = "no alerts yet";
        this.list.appendChild(li);
        return;
      }
      for (const evt of this.recent) {
        const li = document.createElement("li");
        li.className = "al-item " + evt.alert_severity;
        const left = document.createElement("span");
        left.className = "al-sev";
        left.textContent = evt.alert_severity;
        const mid = document.createElement("span");
        mid.className = "al-risk";
        mid.textContent = "risk " + evt.risk_score.toFixed(2);
        const right = document.createElement("span");
        right.className = "al-time";
        right.textContent = fmtClock(evt.timestamp);
        li.appendChild(left);
        li.appendChild(mid);
        li.appendChild(right);
        this.list.appendChild(li);
      }
    }
  }

  function fmtClock(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }

  App.Transcript = Transcript;
  App.Exposure = Exposure;
})(window.App);
