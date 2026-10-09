from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import render
from .detector import DetectionParams, Face, FaceEyeDetector
from .source import FrameSource

WINDOW = "Lab 1 - Haar face & eye detection"
TOAST_SECONDS = 1.6
QUIT_KEYS = {ord("q"), ord("Q"), 27}


@dataclass
class Options:
    source: str = "0"
    width: int = 1280
    height: int = 720
    mirror: bool = True
    eyeglasses: bool = False
    detect_every: int = 1
    snapshot_dir: Path = Path("snapshots")
    output: Path | None = None
    display: bool = True
    max_frames: int = 0
    params: DetectionParams = None


class FpsMeter:
    def __init__(self, window: int = 30) -> None:
        self.samples: deque[float] = deque(maxlen=window)
        self.last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self.samples.append(now - self.last)
        self.last = now
        total = sum(self.samples)
        return len(self.samples) / total if total > 0 else 0.0


class Application:
    def __init__(self, options: Options) -> None:
        self.options = options
        self.detector = FaceEyeDetector(
            eyeglasses=options.eyeglasses,
            params=options.params or DetectionParams(),
        )
        self.mirror = options.mirror
        self.show_hud = True
        self.show_labels = True
        self.paused = False
        self.frozen = False
        self.detect_every = max(1, options.detect_every)
        self.fps = FpsMeter()
        self.faces: list[Face] = []
        self.frame_index = 0
        self.detect_ms = 0.0
        self.toast = ""
        self.toast_until = 0.0
        self.snapshots = 0
        self.writer: cv2.VideoWriter | None = None

    def notify(self, text: str) -> None:
        self.toast = text
        self.toast_until = time.monotonic() + TOAST_SECONDS
        print(text)

    def hud_lines(self) -> list[tuple[str, str]]:
        eyes = sum(len(face.eyes) for face in self.faces)
        p = self.detector.params
        return [
            ("FPS", f"{self.fps_value:5.1f}"),
            ("Detect", f"{self.detect_ms:5.1f} ms  (every {self.detect_every} frame)"),
            ("Faces", f"{len(self.faces)}"),
            ("Eyes", f"{eyes}{'' if self.detector.eyes_enabled else '  [off]'}"),
            ("Cascade", "eyeglasses" if self.options.eyeglasses else "eye"),
            ("Params", f"scale {p.face_scale_factor:.2f}  neighbors {p.face_min_neighbors}"),
            ("Snapshots", f"{self.snapshots}"),
        ]

    def hud_hints(self) -> list[str]:
        return [
            "q/ESC quit",
            "s snapshot",
            "e eyes",
            "m mirror",
            "h hud",
            "l labels",
            "p pause",
            "+/- neighbors",
            "[/] scale",
        ]

    def save_snapshot(self, image: np.ndarray) -> None:
        directory = self.options.snapshot_dir
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = directory / f"lab1_{stamp}.png"
        cv2.imwrite(str(path), image)
        self.snapshots += 1
        self.notify(f"Снимок сохранён: {path}")

    def handle_key(self, key: int, annotated: np.ndarray) -> bool:
        if key in QUIT_KEYS:
            return False
        p = self.detector.params
        if key in (ord("s"), ord("S")):
            self.save_snapshot(annotated)
        elif key in (ord("e"), ord("E")):
            self.detector.eyes_enabled = not self.detector.eyes_enabled
            self.notify(f"Поиск глаз: {'вкл' if self.detector.eyes_enabled else 'выкл'}")
        elif key in (ord("m"), ord("M")):
            self.mirror = not self.mirror
            self.notify(f"Зеркальный режим: {'вкл' if self.mirror else 'выкл'}")
        elif key in (ord("h"), ord("H")):
            self.show_hud = not self.show_hud
        elif key in (ord("l"), ord("L")):
            self.show_labels = not self.show_labels
        elif key in (ord("p"), ord("P"), ord(" ")):
            self.paused = not self.paused
            self.notify("Пауза" if self.paused else "Продолжение")
        elif key in (ord("+"), ord("=")):
            p.face_min_neighbors = min(20, p.face_min_neighbors + 1)
            self.notify(f"minNeighbors = {p.face_min_neighbors}")
        elif key in (ord("-"), ord("_")):
            p.face_min_neighbors = max(1, p.face_min_neighbors - 1)
            self.notify(f"minNeighbors = {p.face_min_neighbors}")
        elif key == ord("]"):
            p.face_scale_factor = round(min(1.5, p.face_scale_factor + 0.01), 2)
            self.notify(f"scaleFactor = {p.face_scale_factor}")
        elif key == ord("["):
            p.face_scale_factor = round(max(1.01, p.face_scale_factor - 0.01), 2)
            self.notify(f"scaleFactor = {p.face_scale_factor}")
        elif key in (ord("c"), ord("C")):
            self.detector.equalize = not self.detector.equalize
            self.notify(f"equalizeHist: {'вкл' if self.detector.equalize else 'выкл'}")
        return True

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        render.draw_faces(annotated, self.faces, self.show_labels)
        if self.show_hud:
            render.draw_hud(annotated, self.hud_lines(), self.hud_hints())
        if not self.faces:
            render.draw_message(annotated, "no faces detected")
        if self.toast and time.monotonic() < self.toast_until:
            render.draw_toast(annotated, self.toast)
        return annotated

    def open_writer(self, source: FrameSource) -> None:
        if self.options.output is None:
            return
        self.options.output.parent.mkdir(parents=True, exist_ok=True)
        width, height = source.size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.options.output), fourcc, source.fps, (width, height)
        )

    def run(self) -> int:
        options = self.options
        self.fps_value = 0.0
        with FrameSource(options.source, options.width, options.height) as source:
            width, height = source.size
            print(f"Источник: {source.title}  {width}x{height}")
            self.open_writer(source)
            if options.display:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, width, height)

            frame = None
            while True:
                if not self.paused or frame is None:
                    incoming = source.read()
                    if incoming is None:
                        print("Поток кадров завершён.")
                        break
                    frame = cv2.flip(incoming, 1) if self.mirror else incoming
                    self.fps_value = self.fps.tick()
                    if self.frame_index % self.detect_every == 0:
                        start = time.perf_counter()
                        self.faces = self.detector.detect(frame)
                        self.detect_ms = (time.perf_counter() - start) * 1000
                    self.frame_index += 1

                annotated = self.annotate(frame)
                if self.writer is not None:
                    self.writer.write(annotated)

                if options.display:
                    cv2.imshow(WINDOW, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self.handle_key(key, annotated):
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break

                if options.max_frames and self.frame_index >= options.max_frames:
                    if not options.display:
                        self.save_snapshot(annotated)
                    break

        if self.writer is not None:
            self.writer.release()
            print(f"Видео сохранено: {options.output}")
        if options.display:
            cv2.destroyAllWindows()
        print(f"Обработано кадров: {self.frame_index}")
        return 0
