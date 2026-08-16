/* Ambient backdrop: a faint, slowly drifting starfield plus a soft ripple that
 * blooms behind the keyboard on each keystroke — just enough to hint that the
 * mic is being listened to, without any loud "AI gradient" mood. Purely
 * decorative; never touches transcript content.
 */
window.App = window.App || {};
(function (App) {
  "use strict";

  class Ambient {
    constructor(canvas) {
      this.c = canvas;
      this.ctx = canvas.getContext("2d");
      this.stars = [];
      this.ripples = [];
      this.t = 0;
      this.warm = 0; // 0 = calm, 1 = critical tint (very subtle)
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
      const count = Math.round((this.w * this.h) / 9000);
      this.stars = [];
      for (let i = 0; i < count; i++) {
        this.stars.push({
          x: Math.random() * this.w,
          y: Math.random() * this.h,
          r: Math.random() * 1.1 + 0.2,
          base: Math.random() * 0.4 + 0.15,
          tw: Math.random() * Math.PI * 2,   // twinkle phase
          sp: Math.random() * 0.6 + 0.2,     // twinkle speed
          drift: Math.random() * 0.04 + 0.01, // gentle upward drift
        });
      }
    }

    setMood(mood) { this.warm = mood === "critical" ? 1 : 0; }

    pulse() {
      this.ripples.push({ r: 0, life: 1 });
      if (this.ripples.length > 8) this.ripples.shift();
    }

    _frame() {
      const { ctx, w, h } = this;
      this.t += 0.016;
      ctx.clearRect(0, 0, w, h);

      // stars
      for (const s of this.stars) {
        s.y -= s.drift;
        if (s.y < -2) { s.y = h + 2; s.x = Math.random() * w; }
        const a = s.base + Math.sin(this.t * s.sp + s.tw) * 0.12;
        const tint = this.warm;
        const R = 235, G = 235 - tint * 60, B = 245 - tint * 90;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${R},${G},${B},${Math.max(0, a)})`;
        ctx.fill();
      }

      // soft keystroke ripples, low and central
      const cx = w * 0.5, cy = h * 0.6;
      for (let i = this.ripples.length - 1; i >= 0; i--) {
        const p = this.ripples[i];
        p.r += 4.5;
        p.life -= 0.02;
        if (p.life <= 0) { this.ripples.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(cx, cy, p.r, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(200,210,240,${p.life * 0.05})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      requestAnimationFrame(() => this._frame());
    }
  }

  App.Ambient = Ambient;
})(window.App);
