/* The data panels.
 *
 *  Transcript — the live "what's being typed" line. The SERVER owns the text
 *  (raw stream, refined by the LLM every few words), so the frontend just renders
 *  whatever `transcript` text arrives. On `commit` (idle auto-log, or the finish
 *  button) the line is captured into the persistent session history on the left.
 *
 *  ControlPanel — the right-hand glass panel. A slider toggles live LLM
 *  correction on/off, plus a little status of what the correction is doing and
 *  where the events are coming from.
 */
window.App = window.App || {};
(function (App) {
  "use strict";

  const LOG_KEY = "session_logs";

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
      this.logs = loadLogs();  // [{ t, text }], newest first
      this._renderHistory();
      this.setText("");
    }

    // Authoritative line from the server (already de-underscored to spaces).
    setText(text) {
      this.typed.textContent = text || "";
      this.typed.scrollTop = this.typed.scrollHeight;
    }

    // Server committed the line (idle or finish) → append to history, clear line.
    commit(text) {
      const clean = (text || "").trim();
      if (clean) {
        this.logs.unshift({ t: Date.now() / 1000, text: clean });
        if (this.logs.length > 100) this.logs.pop();
        saveLogs(this.logs);
        this._renderHistory();
      }
      this.setText("");
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
        li.textContent = "nothing logged yet";
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

  // ---- LLM correction control panel (right) ------------------------------
  class ControlPanel {
    constructor(onToggle) {
      this.onToggle = onToggle;                       // (enabled) => void
      this.panel = document.getElementById("control");
      this.toggle = document.getElementById("llm-toggle");
      this.stateEl = document.getElementById("ctrl-state");
      this.statusEl = document.getElementById("ctrl-status");
      this.noteEl = document.getElementById("ctrl-note");
      this.sourceEl = document.getElementById("ctrl-source");
      this.cadenceEl = document.getElementById("ctrl-cadence");
      this.available = false;
      this.enabled = false;
      this.correctIdle = 5;

      this.toggle.addEventListener("change", () => {
        if (!this.available) { this.toggle.checked = false; return; }
        this.onToggle(this.toggle.checked);           // optimistic; server confirms
      });
      this._render();
    }

    // Authoritative state from the backend (source of truth).
    setLlmState({ enabled, available, correct_idle }) {
      this.available = !!available;
      this.enabled = !!enabled;
      if (typeof correct_idle === "number") this.correctIdle = correct_idle;
      this.toggle.checked = this.enabled;
      this.toggle.disabled = !this.available;
      this._render();
    }

    setSource(mode) {
      const label = { connecting: "connecting…", live: "live", offline: "demo (offline)" }[mode] || mode;
      if (this.sourceEl) this.sourceEl.textContent = label;
    }

    _render() {
      const cadence = this.correctIdle > 0
        ? "on pause (" + Math.round(this.correctIdle) + "s) or finish"
        : "on finish / auto-log";
      this.cadenceEl.textContent = cadence;
      this.panel.classList.toggle("on", this.available && this.enabled);
      this.panel.classList.toggle("off", !this.available || !this.enabled);

      if (!this.available) {
        this.stateEl.textContent = "unavailable";
        this.statusEl.textContent = "raw stream only";
        this.noteEl.textContent = "start the server with --llm to enable correction";
        return;
      }
      if (this.enabled) {
        this.stateEl.textContent = "correcting";
        this.statusEl.textContent = "corrects the whole line when you pause typing";
        this.noteEl.textContent = "";
      } else {
        this.stateEl.textContent = "off";
        this.statusEl.textContent = "showing the raw acoustic decode";
        this.noteEl.textContent = "";
      }
    }
  }

  App.Transcript = Transcript;
  App.ControlPanel = ControlPanel;
})(window.App);
