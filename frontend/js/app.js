/* Bootstrap: build panels, connect the source, route events.
 *
 * Every panel couples only to the event schema. The alarm latches (stays lit until
 * acknowledged with Ctrl+Shift+X, also Cmd+Shift+X on macOS — window-focus only,
 * never a global OS hotkey). Finish captures the current typed text into the log.
 */
(function (App) {
  "use strict";

  const ambient = new App.Ambient(document.getElementById("stars"));
  const keyboard = new App.Keyboard(document.getElementById("keyboard"));
  const transcript = new App.Transcript();
  const exposure = new App.Exposure();

  App._onMood = (mood) => ambient.setMood(mood);

  const sourcePill = document.getElementById("source-pill");
  const sourceText = document.getElementById("source-text");
  function setStatus({ mode }) {
    if (!sourcePill || !sourceText) return;   // source pill removed from the top bar
    sourcePill.className = "source-pill " + mode;
    sourceText.textContent = { connecting: "connecting", live: "live", offline: "demo" }[mode] || mode;
  }

  const source = App.connectSource({
    onStatus: setStatus,
    onState: (state) => exposure.applyState(state),
    onEvent: (evt) => {
      keyboard.hit(evt.key_top1, evt.confidence, evt.alert_severity); // brightness = confidence
      transcript.push(evt);
      exposure.applyEvent(evt);                                        // alarm = risk_score
      ambient.pulse(evt.confidence);
    },
  });

  // Finish → capture typed text into the persistent log.
  document.getElementById("finish").addEventListener("click", () => transcript.finish());
  document.getElementById("clear-logs").addEventListener("click", () => transcript.clearLogs());

  // Keyboard design picker — cycle themes with the glass arrows (persisted).
  const THEMES = [
    { id: "rgb", name: "Aurora RGB" },
    { id: "cloud", name: "Cloud" },
    { id: "lily", name: "Lilypad" },
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

  // Acknowledge / clear the latched alarm. Window-focus only.
  window.addEventListener("keydown", (e) => {
    if (e.shiftKey && (e.ctrlKey || e.metaKey) && (e.key === "X" || e.key === "x")) {
      e.preventDefault();
      source.clearLatch();
    }
  });
})(window.App);
