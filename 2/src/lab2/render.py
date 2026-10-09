from __future__ import annotations

import cv2
import numpy as np

from .matcher import MatchResult, Template

QUAD_COLOR = (60, 60, 240)
KEYPOINT_COLOR = (230, 160, 60)
INLIER_COLOR = (120, 240, 160)
LINE_COLOR = (200, 170, 255)
PANEL_COLOR = (24, 24, 28)
TEXT_COLOR = (240, 240, 240)
ACCENT_COLOR = (120, 200, 255)
WARN_COLOR = (90, 120, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(image: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.62) -> None:
    x, y = max(0, x), max(0, y)
    sub = image[y : y + h, x : x + w]
    if sub.size == 0:
        return
    tint = np.full(sub.shape, PANEL_COLOR, dtype=np.uint8)
    cv2.addWeighted(tint, alpha, sub, 1 - alpha, 0, dst=sub)


def _draw_keypoints(image: np.ndarray, keypoints, offset=(0, 0), color=KEYPOINT_COLOR) -> None:
    ox, oy = offset
    for kp in keypoints:
        x, y = int(kp.pt[0]) + ox, int(kp.pt[1]) + oy
        radius = max(2, int(kp.size / 2))
        cv2.circle(image, (x, y), radius, color, 1, cv2.LINE_AA)


def compose(
    scene: np.ndarray,
    template: Template,
    result: MatchResult,
    show_keypoints: bool = True,
    show_lines: bool = True,
    only_inliers: bool = True,
) -> np.ndarray:
    sh, sw = scene.shape[:2]
    th, tw = template.image.shape[:2]
    scale = min(1.0, (sh * 0.55) / th)
    panel_w, panel_h = int(tw * scale), int(th * scale)
    thumb = cv2.resize(template.image, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((max(sh, panel_h), sw + panel_w, 3), dtype=np.uint8)
    canvas[:sh, :sw] = scene
    canvas[:panel_h, sw : sw + panel_w] = thumb
    offset = (sw, 0)

    if show_keypoints:
        _draw_keypoints(canvas, result.keypoints)
        for kp in template.keypoints:
            x = int(kp.pt[0] * scale) + sw
            y = int(kp.pt[1] * scale)
            cv2.circle(canvas, (x, y), max(2, int(kp.size * scale / 2)), KEYPOINT_COLOR, 1, cv2.LINE_AA)

    if show_lines and result.good:
        flags = result.inlier_flags or [True] * len(result.good)
        for match, inlier in zip(result.good, flags):
            if only_inliers and not inlier:
                continue
            sx, sy = result.keypoints[match.trainIdx].pt
            tx, ty = template.keypoints[match.queryIdx].pt
            start = (int(sx), int(sy))
            end = (int(tx * scale) + offset[0], int(ty * scale))
            cv2.line(canvas, start, end, INLIER_COLOR if inlier else LINE_COLOR, 1, cv2.LINE_AA)
            cv2.circle(canvas, start, 3, INLIER_COLOR if inlier else LINE_COLOR, -1, cv2.LINE_AA)

    cv2.rectangle(canvas, (sw, 0), (sw + panel_w - 1, panel_h - 1), QUAD_COLOR, 2)

    if result.quad is not None:
        points = result.quad.astype(np.int32)
        cv2.polylines(canvas, [points], True, QUAD_COLOR, 3, cv2.LINE_AA)
        corner = points.reshape(-1, 2).min(axis=0)
        _label(canvas, template.name, (int(corner[0]), int(corner[1]) - 4), QUAD_COLOR)

    return canvas


def _label(image: np.ndarray, text: str, origin: tuple[int, int], color) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    y = max(th + 6, y)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 8, y), color, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x + 4, y - 4), FONT, 0.45, (250, 250, 250), 1, cv2.LINE_AA)


def draw_hud(image: np.ndarray, lines: list[tuple[str, str]], hints: list[str]) -> None:
    height, width = image.shape[:2]
    pad, row = 12, 22
    label_width = max(cv2.getTextSize(k, FONT, 0.5, 1)[0][0] for k, _ in lines) + 10
    value_width = max(cv2.getTextSize(v, FONT, 0.5, 1)[0][0] for _, v in lines)
    panel_w = min(width - 2 * pad, label_width + value_width + 2 * pad)
    _panel(image, pad, pad, panel_w, row * len(lines) + pad)
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
        image, text, (max(pad, (width - tw) // 2), height - bar_h + th + 6),
        FONT, scale, TEXT_COLOR, 1, cv2.LINE_AA,
    )


def draw_toast(image: np.ndarray, text: str) -> None:
    height, width = image.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 1)
    x = (width - tw) // 2 - 12
    y = int(height * 0.12)
    _panel(image, x, y, tw + 24, th + 18, 0.75)
    cv2.putText(image, text, (x + 12, y + th + 8), FONT, 0.52, ACCENT_COLOR, 1, cv2.LINE_AA)


def draw_message(image: np.ndarray, text: str) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    height, width = image.shape[:2]
    cv2.putText(
        image, text, ((width - tw) // 2, (height + th) // 2),
        FONT, 0.6, WARN_COLOR, 2, cv2.LINE_AA,
    )
