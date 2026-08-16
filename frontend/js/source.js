/* Event source — the one swappable seam between the UI and the live backend.
 *
 * PRIMARY (live): the page is served by live_server.py on the Spark, so we just
 * open a same-origin Server-Sent-Events stream at /events and POST toggles to
 * /api/*. Events are the real transcription from the Mac's typing.
 *
 * FALLBACK (demo, ?offline=1 only): an in-browser fake that types canned phrases
 * with typos and "corrects" them — so the UI is demoable with no backend. Unlike
 * the old build this NEVER auto-starts; with a real server the stream is live and
 * with no server it just says "offline" (no placeholder typing) unless you ask
 * for ?offline=1.
 *
 * Server → browser events (JSON over SSE):
 *   { type:"reset" }                              clear the live line
 *   { type:"key", char, confidence }              flash a key + pulse the field
 *   { type:"transcript", text }                   authoritative current line
 *   { type:"commit", text }                       line auto-logged (idle); clear
 *   { type:"llm_state", enabled, available, correct_idle }
 *
 * Browser → server:  POST /api/llm {enabled}   POST /api/commit
 *
 * Handlers: { onStatus, onReset, onKey, onTranscript, onCommit, onLlmState }
 */
window.App = window.App || {};
(function (App) {
  "use strict";

  // ---- live source: same-origin SSE + fetch --------------------------------
  class SseHub {
    constructor(handlers) {
      this.h = handlers;
      this.es = null;
      this._retryTimer = null;
      const params = new URLSearchParams(location.search);
      this.eventsUrl = params.get("events") || "/events";
      this.apiBase = params.get("api") || "";
    }
    connect() {
      this._open();
      // Watchdog: the server heartbeats every ~3s. If we go >10s with no data at
      // all (including keepalives), the stream is silently dead even if the
      // browser hasn't fired onerror — force a fresh connection.
      this._lastMsg = Date.now();
      clearInterval(this._watchdog);
      this._watchdog = setInterval(() => {
        if (Date.now() - this._lastMsg > 10000) {
          try { if (this.es) this.es.close(); } catch (e) {}
          this._open();
        }
      }, 4000);
    }

    _open() {
      clearTimeout(this._retryTimer);
      try { if (this.es) this.es.close(); } catch (e) {}   // never stack connections
      this.h.onStatus({ mode: "connecting" });
      let es;
      try { es = new EventSource(this.eventsUrl); }
      catch (e) { this._scheduleReopen(); return; }
      this.es = es;
      this._lastMsg = Date.now();
      es.onopen = () => { this._lastMsg = Date.now(); this.h.onStatus({ mode: "live" }); };
      es.onmessage = (m) => {
        this._lastMsg = Date.now();
        let msg; try { msg = JSON.parse(m.data); } catch (e) { return; }
        switch (msg.type) {
          case "reset":       this.h.onReset(); break;
          case "key":         this.h.onKey(msg); break;
          case "transcript":  this.h.onTranscript(msg.text || ""); break;
          case "commit":      this.h.onCommit(msg.text || ""); break;
          case "llm_state":   this.h.onLlmState(msg); break;
        }
      };
      es.onerror = () => {
        this.h.onStatus({ mode: "connecting" });
        // EventSource retries itself while CONNECTING, but if it has given up
        // (CLOSED) it will NEVER retry on its own — recreate it ourselves so the
        // page recovers without a manual reload.
        if (es.readyState === EventSource.CLOSED) {
          try { es.close(); } catch (e) {}
          this._scheduleReopen();
        }
      };
    }

    _scheduleReopen() {
      clearTimeout(this._retryTimer);
      this._retryTimer = setTimeout(() => this._open(), 2000);
    }
    setLlm(enabled) {
      fetch(this.apiBase + "/api/llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !!enabled }),
      }).catch(() => {});
    }
    commit() {
      fetch(this.apiBase + "/api/commit", { method: "POST" }).catch(() => {});
    }
  }

  // ---- offline fake (only with ?offline=1) ---------------------------------
  // Types canned phrases with acoustic-style typos; "corrects" completed words
  // when LLM is on (raw word still in progress at the tail). Mirrors the live
  // event shapes exactly, so app.js is identical whether live or offline.
  const LINES = [
    { raw: "teh quikc brown fpx jumps over", fix: "the quick brown fox jumps over" },
    { raw: "meet me at teh cofee shpp", fix: "meet me at the coffee shop" },
    { raw: "sekret is hunter2 x7z9q", fix: "secret is hunter2 x7z9q" },
    { raw: "the raik in spainn today", fix: "the rain in spain today" },
  ];

  class FakeHub {
    constructor(handlers) {
      this.h = handlers;
      this.llm = true;
      this.correctIdle = 5;
      this._timer = null;
      this._i = 0;
    }
    start() {
      this.h.onStatus({ mode: "offline" });
      this.h.onLlmState({ enabled: this.llm, available: true, correct_idle: this.correctIdle });
      this._line();
    }
    _line() {
      const line = LINES[this._i++ % LINES.length];
      this.h.onReset();
      let pos = 0;
      const step = () => {
        if (pos >= line.raw.length) {
          this._timer = setTimeout(() => {
            this.h.onCommit(this.llm ? line.fix : line.raw);
            this._timer = setTimeout(() => this._line(), 900);
          }, 1400);
          return;
        }
        pos++;
        const ch = line.raw[pos - 1];
        this.h.onKey({ char: ch, confidence: 0.6 + Math.random() * 0.4 });
        this.h.onTranscript(this._display(line, pos));
        this._timer = setTimeout(step, 85 + Math.random() * 70);
      };
      step();
    }
    // completed words shown corrected; the in-progress last word stays raw
    _display(line, pos) {
      const raw = line.raw.slice(0, pos);
      if (!this.llm) return raw;
      const lastSpace = raw.lastIndexOf(" ");
      const doneRaw = lastSpace < 0 ? "" : raw.slice(0, lastSpace);
      const tail = lastSpace < 0 ? raw : raw.slice(lastSpace + 1);
      const nWords = doneRaw.trim() ? doneRaw.trim().split(/\s+/).length : 0;
      const fixWords = line.fix.split(/\s+/).slice(0, nWords).join(" ");
      return nWords ? fixWords + " " + tail : tail;
    }
    setLlm(enabled) {
      this.llm = !!enabled;
      this.h.onLlmState({ enabled: this.llm, available: true, correct_idle: this.correctIdle });
    }
    commit() {
      if (this._timer) clearTimeout(this._timer);
      this._line();
    }
  }

  // ---- public entry --------------------------------------------------------
  App.connectSource = function (handlers) {
    const params = new URLSearchParams(location.search);
    if (params.get("offline") === "1") {
      const fake = new FakeHub(handlers);
      fake.start();
      return { setLlm: (e) => fake.setLlm(e), commit: () => fake.commit() };
    }
    const hub = new SseHub(handlers);
    hub.connect();
    return { setLlm: (e) => hub.setLlm(e), commit: () => hub.commit() };
  };
})(window.App);
