from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import render
from .matcher import MatchResult, ObjectMatcher
from .source import FrameSource

WINDOW = "Lab 2 - keypoint object recognition"
TOAST_SECONDS = 1.8
QUIT_KEYS = {ord("q"), ord("Q"), 27}


@dataclass
class Options:
    template: Path
    source: str = "0"
    width: int = 960
    height: int = 540
    detector: str = "sift"
    ratio: float = 0.75
    min_inliers: int = 10
    max_features: int = 1500
    detect_every: int = 1
    mirror: bool = True
    snapshot_dir: Path = Path("snapshots")
    output: Path | None = None
    display: bool = True
    max_frames: int = 0


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
        self.matcher = ObjectMatcher(
            kind=options.detector,
            ratio=options.ratio,
            min_inliers=options.min_inliers,
            max_features=options.max_features,
        )
        self.matcher.load_template(options.template)
        self.mirror = options.mirror
        self.show_hud = True
        self.show_keypoints = True
        self.show_lines = True
        self.only_inliers = True
        self.paused = False
        self.detect_every = max(1, options.detect_every)
        self.fps = FpsMeter()
        self.fps_value = 0.0
        self.match_ms = 0.0
        self.result = MatchResult()
        self.frame_index = 0
        self.snapshots = 0
        self.toast = ""
        self.toast_until = 0.0
        self.writer: cv2.VideoWriter | None = None

    def notify(self, text: str) -> None:
        self.toast = text
        self.toast_until = time.monotonic() + TOAST_SECONDS
        print(text)

    def hud_lines(self) -> list[tuple[str, str]]:
        template = self.matcher.template
        status = "FOUND" if self.result.found else "not found"
        return [
            ("FPS", f"{self.fps_value:5.1f}"),
            ("Match", f"{self.match_ms:6.1f} ms  (every {self.detect_every} frame)"),
            ("Detector", f"{self.matcher.kind.upper()}  ratio {self.matcher.ratio:.2f}"),
            ("Template", f"{template.name}  ({len(template.keypoints)} kp)"),
            ("Scene kp", f"{len(self.result.keypoints)}"),
            ("Good", f"{len(self.result.good)}"),
            ("Inliers", f"{self.result.inliers} / min {self.matcher.min_inliers}"),
            ("Object", status),
            ("Snapshots", f"{self.snapshots}"),
        ]

    def hud_hints(self) -> list[str]:
        return [
            "q quit", "s snapshot", "t detector", "k keypoints",
            "l lines", "i inliers", "c crop template", "m mirror",
            "h hud", "p pause", "+/- ratio",
        ]

    def save_snapshot(self, image: np.ndarray) -> None:
        directory = self.options.snapshot_dir
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = directory / f"lab2_{stamp}.png"
        cv2.imwrite(str(path), image)
        self.snapshots += 1
        self.notify(f"Снимок сохранён: {path}")

    def crop_template(self, frame: np.ndarray) -> None:
        roi = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
        x, y, w, h = (int(v) for v in roi)
        if w < 16 or h < 16:
            self.notify("Область слишком мала, образец не изменён")
            return
        try:
            self.matcher.set_template(frame[y : y + h, x : x + w].copy(), "crop")
        except ValueError as error:
            self.notify(str(error))
            return
        self.notify(f"Новый образец: {w}x{h}, {len(self.matcher.template.keypoints)} точек")

    def handle_key(self, key: int, composite: np.ndarray, frame: np.ndarray) -> bool:
        if key in QUIT_KEYS:
            return False
        if key in (ord("s"), ord("S")):
            self.save_snapshot(composite)
        elif key in (ord("t"), ord("T")):
            self.notify(f"Детектор: {self.matcher.cycle_detector().upper()}")
        elif key in (ord("k"), ord("K")):
            self.show_keypoints = not self.show_keypoints
        elif key in (ord("l"), ord("L")):
            self.show_lines = not self.show_lines
        elif key in (ord("i"), ord("I")):
            self.only_inliers = not self.only_inliers
            self.notify(f"Линии: {'только инлаеры' if self.only_inliers else 'все совпадения'}")
        elif key in (ord("c"), ord("C")):
            self.crop_template(frame)
        elif key in (ord("m"), ord("M")):
            self.mirror = not self.mirror
        elif key in (ord("h"), ord("H")):
            self.show_hud = not self.show_hud
        elif key in (ord("p"), ord("P"), ord(" ")):
            self.paused = not self.paused
            self.notify("Пауза" if self.paused else "Продолжение")
        elif key in (ord("+"), ord("=")):
            self.matcher.ratio = round(min(0.95, self.matcher.ratio + 0.05), 2)
            self.notify(f"ratio = {self.matcher.ratio}")
        elif key in (ord("-"), ord("_")):
            self.matcher.ratio = round(max(0.3, self.matcher.ratio - 0.05), 2)
            self.notify(f"ratio = {self.matcher.ratio}")
        return True

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        composite = render.compose(
            frame,
            self.matcher.template,
            self.result,
            self.show_keypoints,
            self.show_lines,
            self.only_inliers,
        )
        if self.show_hud:
            render.draw_hud(composite, self.hud_lines(), self.hud_hints())
        if not self.result.found:
            render.draw_message(composite, "object not found")
        if self.toast and time.monotonic() < self.toast_until:
            render.draw_toast(composite, self.toast)
        return composite

    def open_writer(self, size: tuple[int, int], fps: float) -> None:
        if self.options.output is None:
            return
        self.options.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.options.output), fourcc, fps, size)

    def run(self) -> int:
        options = self.options
        with FrameSource(options.source, options.width, options.height) as source:
            print(f"Источник: {source.title}  {source.size[0]}x{source.size[1]}")
            print(f"Образец: {options.template}  ({len(self.matcher.template.keypoints)} точек)")
            if options.display:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

            frame = None
            sized = False
            while True:
                if not self.paused or frame is None:
                    incoming = source.read()
                    if incoming is None:
                        print("Поток кадров завершён.")
                        break
                    frame = cv2.flip(incoming, 1) if self.mirror and source.is_camera else incoming
                    self.fps_value = self.fps.tick()
                    if self.frame_index % self.detect_every == 0:
                        start = time.perf_counter()
                        self.result = self.matcher.match(frame)
                        self.match_ms = (time.perf_counter() - start) * 1000
                    self.frame_index += 1

                composite = self.annotate(frame)
                if not sized:
                    if options.display:
                        cv2.resizeWindow(WINDOW, composite.shape[1], composite.shape[0])
                    self.open_writer((composite.shape[1], composite.shape[0]), source.fps)
                    sized = True
                if self.writer is not None:
                    self.writer.write(composite)

                if options.display:
                    cv2.imshow(WINDOW, composite)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self.handle_key(key, composite, frame):
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break

                if options.max_frames and self.frame_index >= options.max_frames:
                    if not options.display:
                        self.save_snapshot(composite)
                    break

        if self.writer is not None:
            self.writer.release()
            print(f"Видео сохранено: {options.output}")
        if options.display:
            cv2.destroyAllWindows()
        print(f"Обработано кадров: {self.frame_index}")
        return 0
