from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CASCADE_DIR = Path(__file__).resolve().parent / "cascades"

FACE_CASCADE = "haarcascade_frontalface_default.xml"
EYE_CASCADE = "haarcascade_eye.xml"
EYE_GLASSES_CASCADE = "haarcascade_eye_tree_eyeglasses.xml"

Box = tuple[int, int, int, int]


@dataclass
class Face:
    box: Box
    eyes: list[Box] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2


@dataclass
class DetectionParams:
    face_scale_factor: float = 1.15
    face_min_neighbors: int = 6
    face_min_size_ratio: float = 0.10
    eye_scale_factor: float = 1.08
    eye_min_neighbors: int = 10
    eye_band: tuple[float, float] = (0.15, 0.62)


def _load(path: Path) -> cv2.CascadeClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Файл каскада Хаара не найден: {path}")
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError(f"Не удалось загрузить каскад Хаара: {path}")
    return cascade


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def _suppress(boxes: list[Box], threshold: float = 0.25) -> list[Box]:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(_iou(box, other) < threshold for other in kept):
            kept.append(box)
    return kept


class FaceEyeDetector:
    def __init__(
        self,
        cascade_dir: Path | None = None,
        eyeglasses: bool = False,
        params: DetectionParams | None = None,
    ) -> None:
        directory = cascade_dir or CASCADE_DIR
        self.params = params or DetectionParams()
        self.face_cascade = _load(directory / FACE_CASCADE)
        self.eye_cascade = _load(directory / (EYE_GLASSES_CASCADE if eyeglasses else EYE_CASCADE))
        self.eyes_enabled = True
        self.equalize = True

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray) if self.equalize else gray

    def detect_faces(self, gray: np.ndarray) -> list[Box]:
        p = self.params
        side = min(gray.shape[:2])
        minimum = max(48, int(side * p.face_min_size_ratio))
        found = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=p.face_scale_factor,
            minNeighbors=p.face_min_neighbors,
            minSize=(minimum, minimum),
        )
        return [tuple(int(v) for v in box) for box in found]

    def detect_eyes(self, gray: np.ndarray, face: Box) -> list[Box]:
        p = self.params
        fx, fy, fw, fh = face
        top = fy + int(fh * p.eye_band[0])
        bottom = fy + int(fh * p.eye_band[1])
        roi = gray[top:bottom, fx : fx + fw]
        if roi.size == 0:
            return []
        minimum = max(10, int(fw * 0.10))
        maximum = max(minimum + 1, int(fw * 0.42))
        found = self.eye_cascade.detectMultiScale(
            roi,
            scaleFactor=p.eye_scale_factor,
            minNeighbors=p.eye_min_neighbors,
            minSize=(minimum, minimum),
            maxSize=(maximum, maximum),
        )
        boxes = [(fx + int(x), top + int(y), int(w), int(h)) for x, y, w, h in found]
        return sorted(_suppress(boxes)[:2], key=lambda b: b[0])

    def detect(self, frame: np.ndarray) -> list[Face]:
        gray = self.preprocess(frame)
        faces = [Face(box=box) for box in _suppress(self.detect_faces(gray), 0.4)]
        if self.eyes_enabled:
            for face in faces:
                face.eyes = self.detect_eyes(gray, face.box)
        return sorted(faces, key=lambda f: f.box[0])
