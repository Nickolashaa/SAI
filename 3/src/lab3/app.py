from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import ocr, pipeline, render
from .pipeline import BINARIZATIONS, PipelineParams, PipelineResult

WINDOW = "Lab 3 - contour analysis"
OCR_WINDOW = "Lab 3 - Tesseract OCR"

STAGE_NAMES = (
    "1. Исходное изображение",
    "2. Оттенки серого",
    "3. Бинаризация",
    "4. Удаление шумов",
    "5. Поиск контуров",
    "6. Фильтрация контуров",
    "7. Уравнивание контуров",
    "8. Рамки вокруг контуров",
)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def save(image: np.ndarray, directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = directory / f"{prefix}_{stamp}.png"
    cv2.imwrite(str(path), image)
    print(f"Сохранено: {path}")
    return path


def build_stages(result: PipelineResult, show_groups: bool) -> list[tuple[str, np.ndarray]]:
    kept = [shape.contour for shape in result.shapes]
    return list(
        zip(
            STAGE_NAMES,
            [
                result.original,
                result.gray,
                result.binary,
                result.denoised,
                render.draw_all_contours(result.denoised, [], result.all_contours, blank=True),
                render.draw_all_contours(result.original, result.all_contours, kept),
                render.draw_equalized(result.original, result.shapes),
                render.draw_boxes(result.original, result.shapes, show_groups),
            ],
        )
    )


@dataclass
class ContourOptions:
    image: Path
    params: PipelineParams
    stage: int = 0
    display: bool = True
    save_dir: Path = Path("out")
    save_stages: bool = False
    columns: int = 4
    cell_width: int = 420


class ContourApp:
    def __init__(self, options: ContourOptions) -> None:
        self.options = options
        self.params = options.params
        self.image = read_image(options.image)
        self.stage = options.stage
        self.show_groups = True
        self.result = pipeline.run(self.image, self.params)

    def recompute(self) -> None:
        self.result = pipeline.run(self.image, self.params)

    def summary(self) -> list[str]:
        p = self.params
        return [
            f"Файл: {self.options.image.name}   "
            f"бинаризация: {p.binarization}   порог: {p.threshold}   "
            f"ядро: {p.kernel_size}   мин. площадь: {p.min_area}   "
            f"точек уравнивания: {p.equalize_points}",
            f"Контуров найдено: {len(self.result.all_contours)}   "
            f"после фильтрации: {len(self.result.shapes)}   "
            f"отброшено: {self.result.rejected}   "
            f"групп по форме: {self.result.groups}",
        ]

    def compose(self) -> np.ndarray:
        stages = build_stages(self.result, self.show_groups)
        if self.stage == 0:
            body = render.montage(stages, self.options.columns, self.options.cell_width)
        else:
            caption, image = stages[self.stage - 1]
            body = render.tile(image, caption, (image.shape[1], image.shape[0]))
            body = cv2.copyMakeBorder(body, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=render.BACKGROUND)
        bar = render.info_bar(body.shape[1], self.summary())
        hints = render.info_bar(
            body.shape[1],
            ["q выход | s снимок | 0 сетка | 1-8 этап | b бинаризация | "
             "+/- порог | [/] ядро | ,/. мин. площадь | n точки | g группы | e внешние/все"],
        )
        return np.vstack([bar, body, hints])

    def handle_key(self, key: int, composite: np.ndarray) -> bool:
        p = self.params
        if key in (ord("q"), ord("Q"), 27):
            return False
        if ord("0") <= key <= ord("8"):
            self.stage = key - ord("0")
            return True
        if key in (ord("s"), ord("S")):
            save(composite, self.options.save_dir, "lab3_contours")
            return True
        if key in (ord("b"), ord("B")):
            p.binarization = BINARIZATIONS[(BINARIZATIONS.index(p.binarization) + 1) % 3]
        elif key in (ord("+"), ord("=")):
            p.threshold = min(255, p.threshold + 5)
        elif key in (ord("-"), ord("_")):
            p.threshold = max(0, p.threshold - 5)
        elif key == ord("]"):
            p.kernel_size = min(15, p.kernel_size + 2)
        elif key == ord("["):
            p.kernel_size = max(1, p.kernel_size - 2)
        elif key == ord("."):
            p.min_area = p.min_area + 10
        elif key == ord(","):
            p.min_area = max(0, p.min_area - 10)
        elif key in (ord("n"), ord("N")):
            p.equalize_points = 32 if p.equalize_points >= 128 else p.equalize_points * 2
        elif key in (ord("g"), ord("G")):
            self.show_groups = not self.show_groups
            return True
        elif key in (ord("e"), ord("E")):
            p.external_only = not p.external_only
        else:
            return True
        self.recompute()
        return True

    def run(self) -> int:
        for line in self.summary():
            print(line)
        if self.options.save_stages:
            for index, (caption, image) in enumerate(build_stages(self.result, self.show_groups), 1):
                save(render.as_bgr(image), self.options.save_dir, f"stage{index}")
        composite = self.compose()
        if not self.options.display:
            save(composite, self.options.save_dir, "lab3_contours")
            return 0

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(1800, composite.shape[1]), min(1000, composite.shape[0]))
        while True:
            cv2.imshow(WINDOW, composite)
            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                if not self.handle_key(key, composite):
                    break
                composite = self.compose()
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        return 0


@dataclass
class OcrOptions:
    image: Path
    lang: str = "rus+eng"
    psm: int = 6
    preprocess: bool = True
    min_confidence: float = 40.0
    display: bool = True
    save_dir: Path = Path("out")
    text_out: Path | None = None


class OcrApp:
    def __init__(self, options: OcrOptions) -> None:
        self.options = options
        self.image = read_image(options.image)
        ocr.check_languages(options.lang)
        self.result = ocr.recognize(
            self.image, options.lang, options.psm,
            options.preprocess, options.min_confidence,
        )

    def run(self) -> int:
        result = self.result
        print(f"Язык распознавания: {ocr.LANGUAGES.get(result.lang, result.lang)}")
        print(f"Слов распознано: {len(result.words)}, "
              f"средняя уверенность: {result.mean_confidence:.1f}%")
        print("-" * 60)
        print(result.text)
        print("-" * 60)

        if self.options.text_out:
            self.options.text_out.parent.mkdir(parents=True, exist_ok=True)
            self.options.text_out.write_text(result.text, encoding="utf-8")
            print(f"Текст сохранён: {self.options.text_out}")

        composite = ocr.side_by_side(self.image, result)
        if not self.options.display:
            save(composite, self.options.save_dir, "lab3_ocr")
            return 0

        cv2.namedWindow(OCR_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(OCR_WINDOW, min(1800, composite.shape[1]), min(1000, composite.shape[0]))
        while True:
            cv2.imshow(OCR_WINDOW, composite)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                save(composite, self.options.save_dir, "lab3_ocr")
            if cv2.getWindowProperty(OCR_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        return 0
