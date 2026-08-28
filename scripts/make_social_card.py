#!/usr/bin/env python3
"""Render the GitHub social preview card (assets/social-card.png).

GitHub shows this whenever the repo is linked on Twitter, Slack, Discord or
anywhere else that reads OpenGraph tags. Without one you get an auto-generated
card of the owner's avatar and the description, which is forgettable.

    pip install Pillow
    python scripts/make_social_card.py

1280x640 is GitHub's recommended size. It is displayed small, so this leans on
one number rather than trying to fit the pitch.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "social-card.png"

W, H = 1280, 640
BG = (13, 17, 23)
PANEL = (22, 27, 34)
FG = (237, 242, 247)
DIM = (125, 133, 144)
RED = (248, 113, 113)
GREEN = (74, 222, 128)
BLUE = (96, 165, 250)


def font(size: int, mono: bool = False, bold: bool = False):
    candidates = (
        ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"]
        if mono
        else [
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    mono_xl = font(62, mono=True)
    body = font(27)
    mono_sm = font(20, mono=True)
    mono_xs = font(18, mono=True)

    # Accent rule down the left edge.
    d.rectangle([0, 0, 8, H], fill=BLUE)

    x = 64
    d.text((x, 62), "turnchunk", font=mono_xl, fill=FG)
    d.text((x, 148), "Chunking that understands who is speaking.", font=body, fill=DIM)

    # The contrast, which is the entire argument.
    top, gap, ph = 224, 22, 150
    pw = W - x * 2

    d.rounded_rectangle([x, top, x + pw, top + ph], radius=12, fill=PANEL)
    d.text((x + 26, top + 24), "RecursiveCharacterTextSplitter", font=mono_sm, fill=DIM)
    d.text((x + 26, top + 66), "3 of 10 chunks cannot be attributed to a speaker",
           font=mono_sm, fill=RED)
    d.text((x + 26, top + 102), "0 carry a timestamp", font=mono_xs, fill=RED)

    t2 = top + ph + gap
    d.rounded_rectangle([x, t2, x + pw, t2 + ph], radius=12, fill=PANEL)
    d.text((x + 26, t2 + 24), "turnchunk", font=mono_sm, fill=DIM)
    d.text((x + 26, t2 + 66), "0 of 8 chunks cannot be attributed to a speaker",
           font=mono_sm, fill=GREEN)
    d.text((x + 26, t2 + 102), "8 carry a timestamp", font=mono_xs, fill=GREEN)

    # Footer facts.
    d.text((x, H - 62), "Python + TypeScript", font=mono_xs, fill=DIM)
    d.text((x + 250, H - 62), "zero dependencies", font=mono_xs, fill=DIM)
    d.text((x + 480, H - 62), "MIT", font=mono_xs, fill=DIM)
    d.text((x + 580, H - 62), "pip install turnchunk", font=mono_xs, fill=BLUE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    print(f"{OUT.relative_to(ROOT)}  {W}x{H}  {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
