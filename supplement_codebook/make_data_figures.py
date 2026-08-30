#!/usr/bin/env python3
"""Rebuild the two DATA-BACKED paper figures (glyph_key, decode_strip) from their raws.

WHY THIS FILE EXISTS. These two figures were first built by an inline heredoc that was not kept,
so the figures could not be regenerated -- a Rule 35 violation (an instrument that produces a
load-bearing result is part of the result). Everything drawn here is read from disk at run time:

  glyph map    ane-mutable-adapters/vision/data_codebook_n1/compose.jsonl   (combo id -> letter)
  glyph images ane-mutable-adapters/vision/data_codebook_n1/compose/*.png   (one per symbol)
  decode rows  ane-mutable-weights/supplement_codebook/RESULT_codebook_ane_raw_all.json
  decode input ane-mutable-adapters/vision/data_codebook/compose/*.png

NOTHING IS TYPED IN. The symbol->letter map is DERIVED from the corpus, not hardcoded, so a
regenerated corpus cannot silently disagree with the figure. Model outputs are quoted verbatim
from the raw (only ``` fences stripped and newlines collapsed, so a multi-line emission fits one
line); the truncation marker is visible when it happens.

DECLUTTER (operator direction 2026-08-25): no eyebrow, no title, no footer statistics. The paper's
caption under each figure carries that context already, and duplicating it in the image competes
with the thing the figure is for. What stays is the evidence: glyphs, arrows, letters, emissions.

    python3 make_data_figures.py          write glyph_key.html + decode_strip.html
    python3 make_data_figures.py --render also render both to .png via headless Chrome
"""
import base64, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
N1   = os.path.join(ROOT, "ane-mutable-adapters/vision/data_codebook_n1")
CB   = os.path.join(ROOT, "ane-mutable-adapters/vision/data_codebook")
RAW  = os.path.join(ROOT, "ane-mutable-weights/supplement_codebook/RESULT_codebook_ane_raw_all.json")

INK, MUTED, CORAL, PAPER = "#2d3142", "#4f5d75", "#eb6c36", "#f5f5f5"
FONTS = ("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
         "family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap'>")
SANS = "'Geist',system-ui,-apple-system,sans-serif"
MONO = "'Geist Mono',ui-monospace,monospace"


def b64(path, crop=False):
    """crop=True trims the blank margin around the strokes so the glyphs fill their tile.
    This removes EMPTY canvas only (bounding box of the ink, plus a small pad); no stroke is
    touched and no pixel is rescaled here, so the glyph the reader sees is the glyph the model
    was shown."""
    if crop:
        from PIL import Image
        im = Image.open(path).convert("L")
        # THRESHOLD, not difference-against-white. These renders use a 250 page, not 255, and the
        # strokes are antialiased, so differencing against pure white marks every pixel as ink and
        # the crop silently does nothing (both earlier versions failed this way). A mask of
        # "clearly darker than the page" gives the true stroke extent.
        bbox = im.point(lambda v: 255 if v < 200 else 0).getbbox()
        if bbox:
            pad = 10
            l, t, r, b = bbox
            im = Image.open(path).convert("RGB").crop((max(0, l - pad), max(0, t - pad),
                                                       min(im.width, r + pad),
                                                       min(im.height, b + pad)))
            import io
            buf = io.BytesIO(); im.save(buf, "PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def clean(s, n=34):
    """Verbatim emission, fences stripped and newlines collapsed so it sits on one line."""
    s = s.replace("```json", "").replace("```", "").replace("\t", " ")
    s = " ".join(s.split())
    return (s[:n] + "…") if len(s) > n else s


def glyph_map():
    """DERIVE symbol id -> letter from the single-symbol corpus. Never hardcoded."""
    m, img = {}, {}
    with open(os.path.join(N1, "compose.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if len(r["combo"]) != 1:
                continue
            sid = r["combo"][0]
            m.setdefault(sid, r["target"])
            img.setdefault(sid, r["image"])
            if m[sid] != r["target"]:
                raise SystemExit(f"corpus disagrees on symbol {sid}: {m[sid]} vs {r['target']}")
    if not m:
        raise SystemExit("no single-symbol rows found")
    return m, img


def build_glyph_key():
    m, img = glyph_map()
    ids = sorted(m)
    tiles = []
    for sid in ids:
        p = img[sid]
        if not os.path.exists(p):
            raise SystemExit(f"missing glyph image for symbol {sid}: {p}")
        tiles.append(f"""
      <figure class="cell">
        <img src="{b64(p, crop=True)}" alt="invented glyph {sid}">
        <svg class="arrow" viewBox="0 0 12 26" aria-hidden="true">
          <line x1="6" y1="1" x2="6" y2="18" stroke="{MUTED}" stroke-width="1.2"/>
          <polygon points="2,17 10,17 6,24" fill="{MUTED}"/>
        </svg>
        <figcaption>{m[sid]}</figcaption>
      </figure>""")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>The codebook</title>{FONTS}
<style>
  *{{box-sizing:border-box;}}
  html,body{{margin:0;background:{PAPER};}}
  .frame{{display:flex;justify-content:center;padding:34px 34px;}}
  .row{{display:flex;gap:30px;align-items:flex-start;}}
  .cell{{margin:0;display:flex;flex-direction:column;align-items:center;gap:8px;}}
  .cell img{{width:78px;height:78px;object-fit:contain;background:#fff;border-radius:7px;display:block;padding:11px;}}
  .arrow{{width:12px;height:26px;display:block;}}
  figcaption{{font-family:{SANS};font-size:21px;font-weight:600;color:{INK};line-height:1;}}
</style></head><body><div class="frame"><div class="row">{''.join(tiles)}
</div></div></body></html>"""


def build_decode(n_rows=4):
    rows_raw = json.load(open(RAW))
    base = [r for r in rows_raw if r["arm"] == "BASE(off)"]
    adapt = {r["img"]: r for r in rows_raw if r["arm"] == "ADAPTER(on)"}
    out = []
    for b in base[:n_rows]:
        a = adapt.get(b["img"])
        if a is None:
            raise SystemExit(f"no ADAPTER(on) row for {b['img']}")
        p = os.path.join(CB, "compose", b["img"])
        if not os.path.exists(p):
            raise SystemExit(f"missing decode input image: {p}")
        out.append(f"""
      <div class="r">
        <img src="{b64(p, crop=True)}" alt="glyph string input">
        <svg class="ar" viewBox="0 0 34 10" aria-hidden="true">
          <line x1="0" y1="5" x2="26" y2="5" stroke="{MUTED}" stroke-width="1.1"/>
          <polygon points="25,1.6 25,8.4 32,5" fill="{MUTED}"/>
        </svg>
        <div class="base">{clean(b['got'])}</div>
        <svg class="ar" viewBox="0 0 34 10" aria-hidden="true">
          <line x1="0" y1="5" x2="26" y2="5" stroke="{CORAL}" stroke-width="1.3"/>
          <polygon points="25,1.6 25,8.4 32,5" fill="{CORAL}"/>
        </svg>
        <div class="adapt">{clean(a['got'], 16)}</div>
      </div>""")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Decode</title>{FONTS}
<style>
  *{{box-sizing:border-box;}}
  html,body{{margin:0;background:{PAPER};}}
  .frame{{padding:26px 30px;display:flex;justify-content:center;}}
  table{{border-collapse:collapse;}}
  .hd{{display:grid;grid-template-columns:150px 40px 142px 40px 104px;gap:16px;align-items:center;
       font-family:{MONO};font-size:11px;color:{MUTED};letter-spacing:.06em;margin-bottom:6px;}}
  .hd span{{text-align:center;}}
  .hd span:nth-child(2),.hd span:nth-child(4){{visibility:hidden;}}
  .adapt{{text-align:center;}}
  .r{{display:grid;grid-template-columns:150px 40px 142px 40px 104px;gap:16px;align-items:center;
      padding:13px 0;border-top:1px solid rgba(45,49,66,.10);}}
  .r img{{width:150px;height:auto;background:#fff;border-radius:6px;display:block;padding:7px 6px;}}
  .ar{{width:34px;height:10px;}}
  .base{{font-family:{MONO};font-size:12.5px;color:{MUTED};white-space:nowrap;}}
  .adapt{{font-family:{SANS};font-size:19px;font-weight:600;color:{CORAL};}}
</style></head><body><div class="frame"><div>
    <div class="hd"><span>input</span><span>a</span><span>base model</span><span>b</span><span>+ adapter</span></div>
    {''.join(out)}
</div></div></body></html>"""


def render(name, w, h):
    """Headless Chrome writes the PNG and then does NOT exit on this build, so waiting on the
    process hangs. Spawn it, poll for the file to stop growing, then kill it."""
    import time
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    src, png = os.path.join(HERE, name + ".html"), os.path.join(HERE, name + ".png")
    ud = f"/tmp/chrome-fig-{name}"
    if os.path.exists(png):
        os.remove(png)
    p = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                          f"--user-data-dir={ud}", "--force-device-scale-factor=2",
                          f"--window-size={w},{h}", "--default-background-color=F5F5F5FF",
                          "--virtual-time-budget=5000", f"--screenshot={png}", "file://" + src],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last = -1
    for _ in range(60):                      # up to ~30 s
        time.sleep(0.5)
        if os.path.exists(png):
            size = os.path.getsize(png)
            if size and size == last:        # written and stable
                break
            last = size
    p.kill(); p.wait()
    subprocess.run(["rm", "-rf", ud], capture_output=True)
    if not os.path.exists(png):
        raise SystemExit(f"render produced no file for {name}")
    return png


if __name__ == "__main__":
    m, _ = glyph_map()
    print("  symbol -> letter (derived from corpus):",
          " ".join(f"{k}:{m[k]}" for k in sorted(m)))
    for name, html, size in (("glyph_key", build_glyph_key(), (1000, 200)),
                             ("decode_strip", build_decode(), (620, 430))):
        open(os.path.join(HERE, name + ".html"), "w").write(html)
        print(f"  wrote {name}.html")
        if "--render" in sys.argv:
            print("  rendered", render(name, *size))


def autocrop_png(path, pad=48, page=(245, 245, 245), tol=3):
    """Trim a rendered figure to its content box plus a uniform margin.

    Done on the RASTER rather than by guessing a viewBox: an SVG can carry invisible elements at
    y=0 that make a source-level bound wrong in either direction, and over-cropping would clip
    real content. Anything differing from the #f5f5f5 page by more than `tol` counts as content
    (white tiles included, which is intended). Idempotent: re-running on a cropped file is a
    no-op, so every figure ends up with the same margin.
    """
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, page)
    mask = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > tol else 0)
    box = mask.getbbox()
    if not box:
        return im.size
    l, t, r, b = box
    out = Image.new("RGB", (r - l + 2 * pad, b - t + 2 * pad), page)
    out.paste(im.crop((l, t, r, b)), (pad, pad))
    out.save(path)
    return out.size
