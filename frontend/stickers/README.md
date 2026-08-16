# stickers

Drop your own hand-drawn keycap stickers here as **transparent PNG (or SVG)** files.

The frontend maps five slots by filename (see the sticker rules in `../css/style.css`):

| file            | appears on   |
|-----------------|--------------|
| `sticker1.png`  | the **Q** key |
| `sticker2.png`  | the **G** key |
| `sticker3.png`  | the **P** key |
| `sticker4.png`  | the **Esc** key |
| `sticker5.png`  | the **space bar** |

Nothing shows until a file exists, so you can add them one at a time. Small, roughly
square art works best (they render ~16–20px). To change which key a sticker sits on,
or add more, edit the `.key[data-k="…"]::after` block in `css/style.css` — every key
has a `data-k` attribute (its character; e.g. `data-k="e"`), and structural keys can be
targeted by class (`.key.space`, `.key.wide-s` for Esc, `.key.mod`, etc.).
