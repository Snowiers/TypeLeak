/* Bootstrap: build the panels, connect the live source, route events.
 *
 * The server owns the decoded text and pushes it as `transcript` events; key
 * flashes come as `key` events; idle auto-logging (and the finish button) arrive
 * as `commit`. The LLM toggle round-trips through the backend, which echoes the
 * authoritative `llm_state` back so the switch always reflects the real state.
 */
(function (App) {
  "use strict";

  const ambient = new App.Ambient(document.getElementById("stars"));
  const keyboard = new App.Keyboard(document.getElementById("keyboard"));
  const transcript = new App.Transcript();

  // source pill in the top bar
  const sourcePill = document.getElementById("source-pill");
  const sourceText = document.getElementById("source-text");
  function setStatus({ mode }) {
    if (sourcePill && sourceText) {
      sourcePill.className = "source-pill " + mode;
      sourceText.textContent =
        { connecting: "connecting", live: "live", offline: "demo" }[mode] || mode;
    }
    control.setSource(mode);
  }

  // control panel needs the source to send toggles; source needs the panel to
  // reflect state — wire them after both exist.
  let source = null;
  const control = new App.ControlPanel((enabled) => {
    if (source) source.setLlm(enabled);
  });

  source = App.connectSource({
    onStatus: setStatus,
    onReset: () => transcript.setText(""),
    onKey: (evt) => {
      keyboard.hit(evt.char, evt.confidence, "none");   // brightness = confidence
      ambient.pulse(evt.confidence);
    },
    onTranscript: (text) => transcript.setText(text),
    onCommit: (text) => transcript.commit(text),
    onLlmState: (st) => control.setLlmState(st),
  });

  // finish → ask the server to finalize + log the current line now.
  document.getElementById("finish").addEventListener("click", () => {
    if (source && source.commit) source.commit();
  });
  document.getElementById("clear-logs").addEventListener("click", () => transcript.clearLogs());

  // Keyboard design picker — cycle themes with the glass arrows (persisted).
  const THEMES = [
    { id: "rgb", name: "RGB" },
    { id: "nvidia", name: "NVIDIA" },
    { id: "lily", name: "Lilypad" },
    { id: "blush", name: "Blush" },
    { id: "cloud", name: "Cloud" },
  ];
  const kb = document.getElementById("keyboard");
  const themeLabel = document.getElementById("theme-name");
  let themeIdx = 0;
  try {
    const saved = localStorage.getItem("kb_theme");
    const i = THEMES.findIndex((t) => t.id === saved);
    if (i >= 0) themeIdx = i;
  } catch (e) {}
  function applyTheme() {
    const t = THEMES[themeIdx];
    kb.dataset.theme = t.id;
    document.body.dataset.kbtheme = t.id;   // background glow follows the keyboard
    themeLabel.textContent = t.name;
    try { localStorage.setItem("kb_theme", t.id); } catch (e) {}
  }
  applyTheme();

  // slide the keyboard off-screen, swap the design while it's gone, slide the next in
  const wrap = document.querySelector(".keyboard-wrap");
  let animating = false;
  function switchTheme(dir) {
    if (animating) return;
    animating = true;
    const outX = dir > 0 ? "-125%" : "125%";
    const inX = dir > 0 ? "125%" : "-125%";
    wrap.style.transition = "transform .3s ease-in";
    wrap.style.transform = "translateX(" + outX + ")";
    setTimeout(() => {
      themeIdx = (themeIdx + dir + THEMES.length) % THEMES.length;
      applyTheme();
      wrap.style.transition = "none";
      wrap.style.transform = "translateX(" + inX + ")";
      void wrap.offsetWidth;                 // reflow so the jump takes effect
      wrap.style.transition = "transform .34s ease-out";
      wrap.style.transform = "translateX(0)";
      setTimeout(() => { animating = false; }, 360);
    }, 300);
  }
  document.getElementById("theme-prev").addEventListener("click", () => switchTheme(-1));
  document.getElementById("theme-next").addEventListener("click", () => switchTheme(1));

  // Mouse parallax — feed pointer position to CSS (background, panels, bar drift).
  window.addEventListener("pointermove", (e) => {
    const x = (e.clientX / window.innerWidth - 0.5);
    const y = (e.clientY / window.innerHeight - 0.5);
    document.body.style.setProperty("--px", x.toFixed(3));
    document.body.style.setProperty("--py", y.toFixed(3));
  });
})(window.App);
