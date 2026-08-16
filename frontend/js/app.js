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

  // Acknowledge / clear the latched alarm. Window-focus only.
  window.addEventListener("keydown", (e) => {
    if (e.shiftKey && (e.ctrlKey || e.metaKey) && (e.key === "X" || e.key === "x")) {
      e.preventDefault();
      source.clearLatch();
    }
  });
})(window.App);
