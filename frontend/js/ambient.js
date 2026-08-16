/* Ambient backdrop: a faint starfield far behind, plus a reactive sound-wave band
 * centered behind the keyboard. The waveform always breathes gently (it's listening)
 * and jumps on each keystroke, its energy decaying back down. Soft blue-lavender by
 * default, nudging warm when the alarm is critical. Decorative — never touches content.
 */
window.App = window.App || {};
(function (App) {
  "use strict";

  // waves cycle slowly through blue → purple → pink (HSL hue drift)

  class Ambient {
    constructor(canvas) {
      this.c = canvas;
      this.ctx = canvas.getContext("2d");
      this.stars = [];
      this.ripples = [];
      this.t = 0;
      this.warm = 0;    // 0 calm → 1 critical
      this.energy = 0;  // 0..1 waveform amplitude, decays each frame
      this._resize();
      window.addEventListener("resize", () => this._resize());
      requestAnimationFrame(() => this._frame());
    }

    _resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.w = this.c.clientWidth;
      this.h = this.c.clientHeight;
      this.c.width = this.w * dpr;
      this.c.height = this.h * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this._seed();
    }

    _seed() {
      const count = 0;   // starfield off — the CSS dotted grid provides texture
      this.stars = [];
      for (let i = 0; i < count; i++) {
        this.stars.push({
          x: Math.random() * this.w,
          y: Math.random() * this.h,
          r: Math.random() * 1.0 + 0.2,
          base: Math.random() * 0.35 + 0.12,
          tw: Math.random() * Math.PI * 2,
          sp: Math.random() * 0.6 + 0.2,
          drift: Math.random() * 0.04 + 0.01,
        });
      }
    }

    setMood(mood) { this.warm = mood === "critical" ? 1 : 0; }

    // called per keystroke; confidence (0..1) scales how hard the waveform jumps
    pulse(confidence) {
      const c = Math.max(0, Math.min(1, confidence == null ? 0.6 : confidence));
      this.energy = Math.min(1, this.energy + 0.3 + c * 0.5);
      this.ripples.push({ r: 0, life: 1 });
      if (this.ripples.length > 8) this.ripples.shift();
    }

    // hue for wave line i, drifting over time; stays in blue(220)→pink(330)
    _hue(i) {
      return 260 + Math.sin(this.t * 0.05) * 32 + i * 22 + Math.sin(this.t * 0.08 + i) * 8;
    }

    _frame() {
      const { ctx, w, h } = this;
      this.t += 0.016;
      this.energy *= 0.94;
      ctx.clearRect(0, 0, w, h);

      // distant stars
      for (const s of this.stars) {
        s.y -= s.drift;
        if (s.y < -2) { s.y = h + 2; s.x = Math.random() * w; }
        const a = s.base + Math.sin(this.t * s.sp + s.tw) * 0.1;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(232,236,255,${Math.max(0, a)})`;
        ctx.fill();
      }

      // reactive sound-wave band behind the keyboard — each line its own drifting hue
      const cy = h * 0.58;
      const lines = 3;
      for (let i = 0; i < lines; i++) {
        const hue = this._hue(i);
        const off = (i - 1) * 13;
        const amp = (18 + i * 8) + this.energy * (48 + i * 26);
        const freq = 0.007 + i * 0.0018;
        const speed = this.t * (1.05 + i * 0.28);
        ctx.beginPath();
        for (let x = 0; x <= w; x += 4) {
          const env = 0.35 + 0.65 * Math.sin((x / w) * Math.PI); // gentle edge fade, never to zero
          const y = cy + off
            + Math.sin(x * freq + speed) * amp * env
            + Math.sin(x * freq * 2.1 - speed * 1.3) * amp * 0.3 * env;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        const alpha = 0.3 + this.energy * 0.28;
        ctx.strokeStyle = `hsla(${hue}, 40%, 65%, ${alpha})`;   /* deeper, so it reads on cream */
        ctx.lineWidth = 2.2;
        ctx.shadowColor = `hsla(${hue}, 80%, 52%, 0.4)`;
        ctx.shadowBlur = 9;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // soft keystroke ripples (mid hue)
      const cx = w * 0.5;
      const rHue = this._hue(1);
      for (let i = this.ripples.length - 1; i >= 0; i--) {
        const p = this.ripples[i];
        p.r += 4.5;
        p.life -= 0.02;
        if (p.life <= 0) { this.ripples.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(cx, cy, p.r, 0, Math.PI * 2);
        ctx.strokeStyle = `hsla(${rHue}, 72%, 55%, ${p.life * 0.12})`;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }

      requestAnimationFrame(() => this._frame());
    }
  }

  App.Ambient = Ambient;
})(window.App);
