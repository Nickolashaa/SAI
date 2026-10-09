from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import Application, Options
from .matcher import DETECTORS, ObjectMatcher
from .source import FrameSourceError

__all__ = ["main", "Application", "Options", "ObjectMatcher"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab2",
        description="Лабораторная работа №2: поиск и распознавание объекта по устойчивым "
        "признакам (SIFT и др.) на статичном изображении и в реальном времени с камеры.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("template", type=Path, help="изображение-образец с искомым объектом")
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="статичное изображение или видеофайл для поиска; без него берётся камера",
    )
    parser.add_argument("--camera", type=int, default=0, help="индекс веб-камеры")
    parser.add_argument("--width", type=int, default=960, help="ширина кадра камеры")
    parser.add_argument("--height", type=int, default=540, help="высота кадра камеры")
    parser.add_argument(
        "--detector", choices=DETECTORS, default="sift", help="алгоритм выделения признаков"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.75, help="порог теста Лоу при отборе совпадений"
    )
    parser.add_argument(
        "--min-inliers", type=int, default=10, help="минимум инлаеров RANSAC для признания объекта"
    )
    parser.add_argument(
        "--max-features", type=int, default=1500, help="ограничение числа ключевых точек"
    )
    parser.add_argument(
        "--detect-every", type=int, default=1, help="искать объект раз в N кадров"
    )
    parser.add_argument("--no-mirror", action="store_true", help="не отражать кадр камеры")
    parser.add_argument(
        "--snapshot-dir", type=Path, default=Path("snapshots"), help="папка для снимков"
    )
    parser.add_argument("--output", type=Path, default=None, help="записать результат в видеофайл")
    parser.add_argument("--no-display", action="store_true", help="работа без окна")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="остановиться после N кадров (0 — без предела)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = Options(
        template=args.template,
        source=str(args.scene) if args.scene else str(args.camera),
        width=args.width,
        height=args.height,
        detector=args.detector,
        ratio=args.ratio,
        min_inliers=args.min_inliers,
        max_features=args.max_features,
        detect_every=args.detect_every,
        mirror=not args.no_mirror,
        snapshot_dir=args.snapshot_dir,
        output=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
    )
    try:
        app = Application(options)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    try:
        return app.run()
    except FrameSourceError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
