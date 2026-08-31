#!/usr/bin/env python3
"""Render the `turnchunk viz --compare` demo to an animated GIF.

Regenerate with:

    pip install Pillow
    python scripts/make_demo_gif.py

Deliberately self-contained: no asciinema, no agg, no vhs, no network. The GIF
in the README is produced by running the real CLI and drawing its real ANSI
output, so it cannot drift from what the tool actually prints.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "demo.gif"

# --- terminal geometry -------------------------------------------------------
FONT_SIZE = 15
LINE_H = 21
PAD_X, PAD_TOP = 22, 44
CHROME_H = 34
MAX_COLS = 90

# --- palette (GitHub-dark adjacent, high contrast) ---------------------------
BG = (13, 17, 23)
CHROME = (22, 27, 34)
FG = (201, 209, 217)
DIM = (110, 118, 129)
RED = (248, 113, 113)
GREEN = (74, 222, 128)
YELLOW = (250, 204, 21)
BLUE = (96, 165, 250)
PROMPT = (139, 148, 158)

ANSI = {"31": RED, "32": GREEN, "33": YELLOW, "34": BLUE, "2": DIM, "1": FG, "0": FG}
ESC_RE = re.compile(r"\x1b\[([0-9;]*)m")


def load_font(size: int):
    for path, index in [
        ("/System/Library/Fonts/Menlo.ttc", 0),
        ("/System/Library/Fonts/Monaco.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Andale Mono.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=index)
            except Exception:
                continue
    return ImageFont.load_default()


def parse_ansi(line: str):
    """Split an ANSI-coloured line into (text, colour) runs."""
    runs, pos, colour = [], 0, FG
    for m in ESC_RE.finditer(line):
        if m.start() > pos:
            runs.append((line[pos : m.start()], colour))
        codes = [c for c in m.group(1).split(";") if c]
        if not codes or codes == ["0"]:
            colour = FG
        else:
            for c in codes:
                if c in ANSI:
                    colour = ANSI[c]
        pos = m.end()
    if pos < len(line):
        runs.append((line[pos:], colour))
    return runs


def capture() -> list[str]:
    """Run the real CLI and capture its coloured output."""
    env = dict(os.environ)
    env.pop("NO_COLOR", None)
    cmd = [
        sys.executable, "-m", "turnchunk.cli", "viz",
        str(ROOT / "examples" / "demo.vtt"),
        "--compare", "--target", "230", "--width", str(MAX_COLS), "--color", "always",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"CLI failed:\n{r.stderr}")
    return r.stdout.rstrip("\n").split("\n")


def main() -> None:
    lines = capture()
    font = load_font(FONT_SIZE)

    probe = Image.new("RGB", (10, 10))
    char_w = ImageDraw.Draw(probe).textlength("M", font=font)

    cmd_text = "turnchunk viz meeting.vtt --compare"
    header = [("$ ", PROMPT), (cmd_text, FG)]

    # The "who said this?" marker is printed past the box edge, so the canvas
    # needs headroom beyond the terminal width or it gets clipped.
    longest = max(len(ESC_RE.sub("", ln)) for ln in lines)
    cols = max(MAX_COLS + 2, longest + 2)
    width = int(PAD_X * 2 + char_w * cols)
    height = int(PAD_TOP + LINE_H * (len(lines) + 1) + 16)

    def base_frame():
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, CHROME_H], fill=CHROME)
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([PAD_X + i * 20, 12, PAD_X + i * 20 + 11, 23], fill=c)
        d.text((width // 2 - 38, 9), "turnchunk", font=font, fill=DIM)
        return img, d

    def draw_line(d, y, runs):
        x = PAD_X
        for text, colour in runs:
            if not text:
                continue
            d.text((x, y), text, font=font, fill=colour)
            x += char_w * len(text)

    frames, durations = [], []

    # 1. type the command
    for n in range(0, len(cmd_text) + 1, 3):
        img, d = base_frame()
        draw_line(d, PAD_TOP, [("$ ", PROMPT), (cmd_text[:n], FG), ("█", GREEN)])
        frames.append(img)
        durations.append(38)

    img, d = base_frame()
    draw_line(d, PAD_TOP, header)
    frames.append(img)
    durations.append(420)

    # 2. reveal the output line by line
    for i in range(1, len(lines) + 1):
        img, d = base_frame()
        draw_line(d, PAD_TOP, header)
        for j, line in enumerate(lines[:i]):
            draw_line(d, PAD_TOP + LINE_H * (j + 1), parse_ansi(line))
        frames.append(img)
        # Pause on the two summary lines - they are the point of the demo.
        stripped = ESC_RE.sub("", lines[i - 1])
        durations.append(1100 if "cannot be attributed" in stripped else 62)

    # 3. hold the finished frame
    frames.append(frames[-1])
    durations.append(4200)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    quantised = [
        f.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for f in frames
    ]
    quantised[0].save(
        OUT,
        save_all=True,
        append_images=quantised[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    kb = OUT.stat().st_size / 1024
    print(f"{OUT.relative_to(ROOT)}  {width}x{height}  {len(frames)} frames  {kb:.0f} KB")
    if kb > 3000:
        print("warning: over 3 MB, GitHub may be slow to load it")


if __name__ == "__main__":
    main()
