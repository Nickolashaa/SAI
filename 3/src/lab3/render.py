from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contours import Shape, to_points

FONT_CANDIDATES = (
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

BACKGROUND = (28, 28, 32)
CAPTION_BG = (44, 44, 52)
CAPTION_FG = (238, 238, 238)
ACCENT = (255, 200, 120)
BOX_COLOR = (90, 230, 120)
REJECTED_COLOR = (110, 110, 130)
EQUALIZED_COLOR = (255, 170, 80)
POINT_COLOR = (120, 200, 255)

GROUP_COLORS = (
    (96, 220, 130), (240, 160, 80), (120, 190, 255), (210, 130, 240),
    (100, 230, 230), (250, 120, 150), (170, 230, 110), (255, 220, 120),
)


def _system_font() -> str | None:
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}", "DejaVu Sans"],
            capture_output=True, text=True, timeout=3,
        )
        path = out.stdout.strip()
        if path and Path(path).exists():
            return path
    except (OSError, subprocess.SubprocessError):
        pass
    return None


@lru_cache(maxsize=16)
def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    path = _system_font()
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def text_size(text: str, size: int) -> tuple[int, int]:
    font = load_font(size)
    box = font.getbbox(text)
    return box[2] - box[0], box[3] - box[1]


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    size: int = 18,
    color: tuple[int, int, int] = CAPTION_FG,
) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(origin, text, font=load_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def put_lines(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    size: int = 18,
    color: tuple[int, int, int] = CAPTION_FG,
    spacing: int = 6,
) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = load_font(size)
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=font, fill=color)
        y += size + spacing
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def draw_all_contours(
    base: np.ndarray, contours: list, kept: list, blank: bool = False
) -> np.ndarray:
    canvas = np.full((*base.shape[:2], 3), BACKGROUND, np.uint8) if blank else as_bgr(base)
    cv2.drawContours(canvas, contours, -1, REJECTED_COLOR, 1, cv2.LINE_AA)
    cv2.drawContours(canvas, kept, -1, BOX_COLOR, 1, cv2.LINE_AA)
    return canvas


def draw_equalized(base: np.ndarray, shapes: list[Shape]) -> np.ndarray:
    canvas = as_bgr(base)
    for shape in shapes:
        points = to_points(shape.equalized)
        cv2.polylines(canvas, [points], True, EQUALIZED_COLOR, 1, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(canvas, (int(x), int(y)), 1, POINT_COLOR, -1, cv2.LINE_AA)
    return canvas


def draw_boxes(base: np.ndarray, shapes: list[Shape], show_groups: bool = True) -> np.ndarray:
    canvas = as_bgr(base)
    for shape in shapes:
        x, y, w, h = shape.box
        color = GROUP_COLORS[shape.group % len(GROUP_COLORS)] if show_groups else BOX_COLOR
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    return canvas


def _caption(width: int, text: str, height: int = 30) -> np.ndarray:
    strip = np.full((height, width, 3), CAPTION_BG, dtype=np.uint8)
    return put_text(strip, text, (10, 6), 17, CAPTION_FG)


def tile(image: np.ndarray, caption: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scaled = cv2.resize(as_bgr(image), (width, height), interpolation=cv2.INTER_AREA)
    return np.vstack([_caption(width, caption), scaled])


def montage(
    stages: list[tuple[str, np.ndarray]], columns: int = 4, cell_width: int = 420
) -> np.ndarray:
    first = stages[0][1]
    ratio = first.shape[0] / first.shape[1]
    cell_height = int(cell_width * ratio)
    tiles = [tile(image, caption, (cell_width, cell_height)) for caption, image in stages]
    blank = np.full_like(tiles[0], BACKGROUND, dtype=np.uint8)
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    return cv2.copyMakeBorder(grid, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=BACKGROUND)


def wrap(lines: list[str], width: int, size: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        current = ""
        for word in line.split(" "):
            candidate = f"{current} {word}" if current else word
            if text_size(candidate, size)[0] <= width or not current:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped


def info_bar(width: int, lines: list[str], size: int = 17) -> np.ndarray:
    rows = wrap(lines, width - 28, size)
    bar = np.full(((size + 7) * len(rows) + 14, width, 3), BACKGROUND, dtype=np.uint8)
    return put_lines(bar, rows, (14, 8), size, ACCENT, spacing=7)
