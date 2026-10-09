from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FrameSourceError(RuntimeError):
    pass


class FrameSource:
    def __init__(self, spec: str, width: int = 0, height: int = 0, loop: bool = True) -> None:
        self.spec = spec
        self.loop = loop
        self.is_camera = False
        self.still: np.ndarray | None = None
        self.capture: cv2.VideoCapture | None = None

        path = Path(spec)
        if spec.isdigit():
            self._open_camera(int(spec), width, height)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            self._open_image(path)
        else:
            self._open_file(path)

    def _open_camera(self, index: int, width: int, height: int) -> None:
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise FrameSourceError(
                f"Не удалось открыть веб-камеру с индексом {index}. "
                "Проверьте подключение камеры или укажите другой индекс через --camera, "
                "либо файл через --source."
            )
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = capture
        self.is_camera = True

    def _open_image(self, path: Path) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FrameSourceError(f"Не удалось прочитать изображение: {path}")
        self.still = image

    def _open_file(self, path: Path) -> None:
        if not path.exists():
            raise FrameSourceError(f"Файл не найден: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FrameSourceError(f"Не удалось открыть видеофайл: {path}")
        self.capture = capture

    @property
    def size(self) -> tuple[int, int]:
        if self.still is not None:
            h, w = self.still.shape[:2]
            return w, h
        assert self.capture is not None
        return (
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self) -> float:
        if self.still is not None:
            return 25.0
        assert self.capture is not None
        value = self.capture.get(cv2.CAP_PROP_FPS)
        return value if 1.0 < value < 240.0 else 25.0

    @property
    def title(self) -> str:
        if self.is_camera:
            return f"camera {self.spec}"
        return Path(self.spec).name

    def read(self) -> np.ndarray | None:
        if self.still is not None:
            return self.still.copy()
        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok and not self.is_camera and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        return frame if ok else None

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
