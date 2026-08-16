/* A Mac keyboard replica (US layout): function row, number row with shifted
 * legends, QWERTY, modifiers with their glyphs, wide space bar, and the inverted-T
 * arrow cluster. Only lowercase letters and space carry a `code` and can light up
 * (that's the model side's VOCAB) — every other key is there purely for the replica.
 *
 * Plain DOM + CSS. No three.js — this is already the low-risk path from CLAUDE.md.
 *
 * A key spec: { main, sub?, code?, cls?, flex? }
 *   main  bottom/primary legend        sub  small upper legend (Mac dual legends)
 *   code  char that flashes on a hit   cls  extra class(es)   flex  width weight
 */
window.App = window.App || {};
(function (App) {
  "use strict";

  const F = (n) => ({ main: "F" + n, cls: "fkey", flex: 1 });
  const letter = (ch) => ({ main: ch.toUpperCase(), code: ch, flex: 1 });
  const dual = (main, sub, flex) => ({ main, sub, flex: flex || 1 });

  const LAYOUT = [
    // function row (shorter)
    {
      cls: "row-fn",
      keys: [
        { main: "esc", cls: "wide-s", flex: 1.4 },
        F(1), F(2), F(3), F(4), F(5), F(6),
        F(7), F(8), F(9), F(10), F(11), F(12),
        { main: "⏻", cls: "fkey", flex: 1 },
      ],
    },
    // number row
    {
      keys: [
        dual("`", "~"),
        dual("1", "!"), dual("2", "@"), dual("3", "#"), dual("4", "$"),
        dual("5", "%"), dual("6", "^"), dual("7", "&"), dual("8", "*"),
        dual("9", "("), dual("0", ")"), dual("-", "_"), dual("=", "+"),
        { main: "delete", cls: "wide r", flex: 1.6 },
      ],
    },
    // tab row
    {
      keys: [
        { main: "⇥", cls: "wide l", flex: 1.5 },
        ...["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"].map(letter),
        dual("[", "{"), dual("]", "}"), dual("\\", "|", 1.4),
      ],
    },
    // caps row
    {
      keys: [
        { main: "⇪", cls: "wide l", flex: 1.75 },
        ...["a", "s", "d", "f", "g", "h", "j", "k", "l"].map(letter),
        dual(";", ":"), dual("'", "\""),
        { main: "return", cls: "wide r", flex: 1.9 },
      ],
    },
    // shift row
    {
      keys: [
        { main: "⇧", cls: "wide l", flex: 2.3 },
        ...["z", "x", "c", "v", "b", "n", "m"].map(letter),
        dual(",", "<"), dual(".", ">"), dual("/", "?"),
        { main: "⇧", cls: "wide r", flex: 2.3 },
      ],
    },
    // bottom row (space + modifiers + arrows)
    {
      keys: [
        { main: "fn", cls: "mod", flex: 1 },
        { main: "⌃", cls: "mod", flex: 1 },
        { main: "⌥", cls: "mod", flex: 1 },
        { main: "⌘", cls: "mod", flex: 1.25 },
        { main: "", code: " ", cls: "space", flex: 5 },
        { main: "⌘", cls: "mod", flex: 1.25 },
        { main: "⌥", cls: "mod", flex: 1 },
        { arrows: true, flex: 3 },
      ],
    },
  ];

  class Keyboard {
    constructor(root) {
      this.root = root;
      this.map = {};
      this._timers = {};
      this._build();
    }

    _build() {
      LAYOUT.forEach((row, r) => {
        const rowEl = document.createElement("div");
        rowEl.className = "kb-row" + (row.cls ? " " + row.cls : "");
        row.keys.forEach((spec, c) => {
          const el = spec.arrows ? this._arrows(spec) : this._key(spec);
          // stagger the RGB underglow diagonally so it sweeps across the board
          el.style.animationDelay = (-(r * 0.5 + c * 0.28)).toFixed(2) + "s";
          rowEl.appendChild(el);
        });
        this.root.appendChild(rowEl);
      });
    }

    _key(spec) {
      const el = document.createElement("div");
      el.className = "key" + (spec.cls ? " " + spec.cls : "");
      el.style.flex = String(spec.flex || 1);
      if (spec.sub) {
        const sub = document.createElement("span");
        sub.className = "sub";
        sub.textContent = spec.sub;
        const main = document.createElement("span");
        main.className = "main";
        main.textContent = spec.main;
        el.appendChild(sub);
        el.appendChild(main);
      } else {
        el.textContent = spec.main;
      }
      if (spec.code !== undefined) {
        this.map[spec.code] = el;
        el.dataset.k = spec.code;   // lets CSS put cute stickers on specific keys
      }
      return el;
    }

    // inverted-T arrow cluster: ◀ then a half-height ▲/▼ column, then ▶
    _arrows(spec) {
      const wrap = document.createElement("div");
      wrap.className = "arrows";
      wrap.style.flex = String(spec.flex || 3);

      const left = document.createElement("div");
      left.className = "key arrow";
      left.textContent = "◀";

      const col = document.createElement("div");
      col.className = "arrow-col";
      const up = document.createElement("div");
      up.className = "key arrow half";
      up.textContent = "▲";
      const down = document.createElement("div");
      down.className = "key arrow half";
      down.textContent = "▼";
      col.appendChild(up);
      col.appendChild(down);

      const right = document.createElement("div");
      right.className = "key arrow";
      right.textContent = "▶";

      wrap.appendChild(left);
      wrap.appendChild(col);
      wrap.appendChild(right);
      return wrap;
    }

    // Flash a predicted key. confidence 0..1 -> brightness; severity -> tint.
    hit(char, confidence, severity) {
      const el = this.map[char];
      if (!el) return;
      const conf = Math.max(0, Math.min(1, confidence || 0));
      el.style.setProperty("--i", conf.toFixed(3));
      el.setAttribute("data-sev", severity || "none");
      el.classList.remove("hit");
      void el.offsetWidth;
      el.classList.add("hit");
      clearTimeout(this._timers[char]);
      this._timers[char] = setTimeout(() => el.classList.remove("hit"), 460);
    }
  }

  App.Keyboard = Keyboard;
})(window.App);
