from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytesseract

from .app import ContourApp, ContourOptions, OcrApp, OcrOptions
from .ocr import OcrUnavailable
from .pipeline import BINARIZATIONS, PipelineParams

__all__ = ["main", "ContourApp", "OcrApp"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab3",
        description="Лабораторная работа №3: контурный анализ изображения с текстом "
        "и распознавание печатного текста через Tesseract OCR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    contours = subparsers.add_parser(
        "contours",
        help="часть 1: пайплайн контурного анализа",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    contours.add_argument("image", type=Path, help="изображение с текстом")
    contours.add_argument(
        "--binarization", choices=BINARIZATIONS, default="otsu", help="способ бинаризации"
    )
    contours.add_argument("--threshold", type=int, default=127, help="порог для fixed")
    contours.add_argument("--block-size", type=int, default=31, help="окно для adaptive")
    contours.add_argument("--constant", type=int, default=10, help="константа C для adaptive")
    contours.add_argument("--kernel-size", type=int, default=3, help="размер морфологического ядра")
    contours.add_argument("--open-iterations", type=int, default=1, help="итераций «открытия»")
    contours.add_argument("--close-iterations", type=int, default=1, help="итераций «закрытия»")
    contours.add_argument("--min-area", type=int, default=30, help="минимальная площадь контура")
    contours.add_argument(
        "--max-area-ratio", type=float, default=0.35, help="максимальная доля площади кадра"
    )
    contours.add_argument("--min-side", type=int, default=4, help="минимальная сторона рамки")
    contours.add_argument(
        "--equalize-points", type=int, default=64, help="длина вектор-контура при уравнивании"
    )
    contours.add_argument(
        "--group-threshold", type=float, default=0.95, help="порог схожести форм по ВКФ"
    )
    contours.add_argument(
        "--all-contours", action="store_true", help="искать и вложенные контуры, не только внешние"
    )
    contours.add_argument("--stage", type=int, default=0, help="показать один этап (0 — сетка)")
    contours.add_argument("--columns", type=int, default=4, help="столбцов в сетке этапов")
    contours.add_argument("--cell-width", type=int, default=420, help="ширина ячейки сетки")
    contours.add_argument("--save-stages", action="store_true", help="сохранить этапы по отдельности")
    contours.add_argument("--save-dir", type=Path, default=Path("out"), help="папка результатов")
    contours.add_argument("--no-display", action="store_true", help="без окна, только сохранение")

    recognize = subparsers.add_parser(
        "ocr",
        help="часть 2: распознавание текста через Tesseract",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    recognize.add_argument("image", type=Path, help="изображение с печатным текстом")
    recognize.add_argument(
        "--lang", default="rus+eng", help="язык: rus, eng или rus+eng"
    )
    recognize.add_argument("--psm", type=int, default=6, help="режим сегментации страницы Tesseract")
    recognize.add_argument(
        "--no-preprocess", action="store_true", help="без предобработки перед распознаванием"
    )
    recognize.add_argument(
        "--min-confidence", type=float, default=40.0, help="порог уверенности для слова"
    )
    recognize.add_argument("--text-out", type=Path, default=None, help="сохранить текст в файл")
    recognize.add_argument("--save-dir", type=Path, default=Path("out"), help="папка результатов")
    recognize.add_argument("--no-display", action="store_true", help="без окна, только сохранение")
    recognize.add_argument(
        "--tesseract-cmd", type=Path, default=None, help="путь к исполняемому файлу tesseract"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "contours":
            params = PipelineParams(
                binarization=args.binarization,
                threshold=args.threshold,
                block_size=args.block_size,
                constant=args.constant,
                kernel_size=args.kernel_size,
                open_iterations=args.open_iterations,
                close_iterations=args.close_iterations,
                min_area=args.min_area,
                max_area_ratio=args.max_area_ratio,
                min_side=args.min_side,
                equalize_points=args.equalize_points,
                group_threshold=args.group_threshold,
                external_only=not args.all_contours,
            )
            app = ContourApp(
                ContourOptions(
                    image=args.image,
                    params=params,
                    stage=args.stage,
                    display=not args.no_display,
                    save_dir=args.save_dir,
                    save_stages=args.save_stages,
                    columns=args.columns,
                    cell_width=args.cell_width,
                )
            )
            return app.run()

        if args.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = str(args.tesseract_cmd)
        app = OcrApp(
            OcrOptions(
                image=args.image,
                lang=args.lang,
                psm=args.psm,
                preprocess=not args.no_preprocess,
                min_confidence=args.min_confidence,
                display=not args.no_display,
                save_dir=args.save_dir,
                text_out=args.text_out,
            )
        )
        return app.run()
    except (FileNotFoundError, OcrUnavailable, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
