from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import Application, Options
from .detector import DetectionParams
from .source import FrameSourceError

__all__ = ["main", "Application", "Options", "DetectionParams"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab1",
        description="Лабораторная работа №1: поиск лиц и глаз в реальном времени "
        "с веб-камеры средствами OpenCV (каскады Хаара).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="индекс веб-камеры")
    parser.add_argument(
        "--source",
        default=None,
        help="путь к изображению или видеофайлу вместо веб-камеры",
    )
    parser.add_argument("--width", type=int, default=1280, help="запрашиваемая ширина кадра")
    parser.add_argument("--height", type=int, default=720, help="запрашиваемая высота кадра")
    parser.add_argument(
        "--no-mirror", action="store_true", help="не отражать кадр по горизонтали"
    )
    parser.add_argument(
        "--eyeglasses",
        action="store_true",
        help="каскад haarcascade_eye_tree_eyeglasses.xml (устойчивее к очкам)",
    )
    parser.add_argument(
        "--no-eyes", action="store_true", help="искать только лица, без глаз"
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="запускать детектор раз в N кадров (ускоряет работу)",
    )
    parser.add_argument("--scale-factor", type=float, default=1.15, help="scaleFactor для лиц")
    parser.add_argument("--min-neighbors", type=int, default=6, help="minNeighbors для лиц")
    parser.add_argument(
        "--eye-scale-factor", type=float, default=1.08, help="scaleFactor для глаз"
    )
    parser.add_argument(
        "--eye-min-neighbors", type=int, default=10, help="minNeighbors для глаз"
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, default=Path("snapshots"), help="папка для снимков"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="записать размеченное видео в файл"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="без окна (для записи в файл или автоматических снимков)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="остановиться после N кадров (0 — без ограничения)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = Options(
        source=args.source if args.source else str(args.camera),
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        eyeglasses=args.eyeglasses,
        detect_every=args.detect_every,
        snapshot_dir=args.snapshot_dir,
        output=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
        params=DetectionParams(
            face_scale_factor=args.scale_factor,
            face_min_neighbors=args.min_neighbors,
            eye_scale_factor=args.eye_scale_factor,
            eye_min_neighbors=args.eye_min_neighbors,
        ),
    )
    try:
        app = Application(options)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    if args.no_eyes:
        app.detector.eyes_enabled = False
    try:
        return app.run()
    except FrameSourceError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
