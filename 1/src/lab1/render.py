from __future__ import annotations

import cv2
import numpy as np

from .detector import Box, Face

FACE_COLOR = (90, 230, 120)
EYE_COLOR = (255, 200, 60)
PANEL_COLOR = (24, 24, 28)
TEXT_COLOR = (240, 240, 240)
ACCENT_COLOR = (120, 200, 255)
WARN_COLOR = (90, 120, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _corner_frame(image: np.ndarray, box: Box, color, thickness: int = 2) -> None:
    x, y, w, h = box
    arm = max(8, min(w, h) // 5)
    for px, py, dx, dy in (
        (x, y, 1, 1),
        (x + w, y, -1, 1),
        (x, y + h, 1, -1),
        (x + w, y + h, -1, -1),
    ):
        cv2.line(image, (px, py), (px + dx * arm, py), color, thickness, cv2.LINE_AA)
        cv2.line(image, (px, py), (px, py + dy * arm), color, thickness, cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)


def _label(image: np.ndarray, text: str, origin: tuple[int, int], color) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    y = max(th + 6, y)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 8, y), color, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x + 4, y - 4), FONT, 0.45, (16, 16, 16), 1, cv2.LINE_AA)


def draw_faces(image: np.ndarray, faces: list[Face], show_labels: bool = True) -> None:
    for index, face in enumerate(faces, start=1):
        _corner_frame(image, face.box, FACE_COLOR, 2)
        if show_labels:
            x, y, _, _ = face.box
            _label(image, f"Face {index} | eyes: {len(face.eyes)}", (x, y - 4), FACE_COLOR)
        for eye in face.eyes:
            ex, ey, ew, eh = eye
            center = (ex + ew // 2, ey + eh // 2)
            axes = (max(3, ew // 2), max(3, eh // 3))
            cv2.ellipse(image, center, axes, 0, 0, 360, EYE_COLOR, 2, cv2.LINE_AA)
            cv2.circle(image, center, 2, EYE_COLOR, -1, cv2.LINE_AA)


def _panel(image: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.62) -> None:
    sub = image[y : y + h, x : x + w]
    if sub.size == 0:
        return
    tint = np.full(sub.shape, PANEL_COLOR, dtype=np.uint8)
    cv2.addWeighted(tint, alpha, sub, 1 - alpha, 0, dst=sub)


def draw_hud(image: np.ndarray, lines: list[tuple[str, str]], hints: list[str]) -> None:
    height, width = image.shape[:2]
    pad, row = 12, 22
    label_width = max(cv2.getTextSize(f"{k}", FONT, 0.5, 1)[0][0] for k, _ in lines) + 10
    value_width = max(cv2.getTextSize(f"{v}", FONT, 0.5, 1)[0][0] for _, v in lines)
    panel_w = min(width - 2 * pad, label_width + value_width + 2 * pad)
    panel_h = row * len(lines) + pad
    _panel(image, pad, pad, panel_w, panel_h)
    for index, (key, value) in enumerate(lines):
        y = pad + row * (index + 1) - 6
        cv2.putText(image, key, (pad + 10, y), FONT, 0.5, ACCENT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(image, value, (pad + 10 + label_width, y), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    if not hints:
        return
    text = "   ".join(hints)
    scale = 0.44
    while scale > 0.28:
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
        if tw <= width - 2 * pad:
            break
        scale -= 0.02
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    bar_h = th + 16
    _panel(image, 0, height - bar_h, width, bar_h, 0.7)
    cv2.putText(
        image,
        text,
        (max(pad, (width - tw) // 2), height - bar_h + th + 6),
        FONT,
        scale,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_toast(image: np.ndarray, text: str) -> None:
    height, width = image.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 1)
    x = (width - tw) // 2 - 12
    y = int(height * 0.12)
    _panel(image, x, y, tw + 24, th + 18, 0.75)
    cv2.putText(image, text, (x + 12, y + th + 8), FONT, 0.52, ACCENT_COLOR, 1, cv2.LINE_AA)


def draw_message(image: np.ndarray, text: str) -> None:
    height, width = image.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    cv2.putText(
        image,
        text,
        ((width - tw) // 2, (height + th) // 2),
        FONT,
        0.6,
        WARN_COLOR,
        2,
        cv2.LINE_AA,
    )
