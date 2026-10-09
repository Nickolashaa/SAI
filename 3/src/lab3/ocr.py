from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from pytesseract import Output, TesseractNotFoundError

from . import render

LANGUAGES = {"rus": "русский", "eng": "английский", "rus+eng": "русский и английский"}

BOX_COLOR = (90, 230, 120)
LOW_CONF_COLOR = (90, 160, 250)
TEXT_COLOR = (255, 230, 170)
PANEL_BG = (28, 28, 32)

INSTALL_HINT = (
    "Tesseract OCR не найден в системе. Установите его и языковые пакеты:\n"
    "  Arch/CachyOS: sudo pacman -S tesseract tesseract-data-rus tesseract-data-eng\n"
    "  Debian/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng\n"
    "  Windows: установщик с https://github.com/UB-Mannheim/tesseract/wiki, "
    "затем указать путь через --tesseract-cmd"
)


class OcrUnavailable(RuntimeError):
    pass


@dataclass
class Word:
    text: str
    box: tuple[int, int, int, int]
    confidence: float
    line: int


@dataclass
class OcrResult:
    text: str
    lang: str
    words: list[Word] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        values = [w.confidence for w in self.words]
        return sum(values) / len(values) if values else 0.0

    def lines(self) -> list[str]:
        return [line for line in self.text.splitlines() if line.strip()]


def available_languages() -> list[str]:
    try:
        return list(pytesseract.get_languages(config=""))
    except TesseractNotFoundError as error:
        raise OcrUnavailable(INSTALL_HINT) from error


def check_languages(lang: str) -> None:
    installed = available_languages()
    missing = [part for part in lang.split("+") if part not in installed]
    if missing:
        raise OcrUnavailable(
            f"Не установлены языковые пакеты Tesseract: {', '.join(missing)}. "
            f"Доступны: {', '.join(sorted(installed))}"
        )


def preprocess(image: np.ndarray, upscale: float = 2.0) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def recognize(
    image: np.ndarray,
    lang: str = "rus+eng",
    psm: int = 6,
    use_preprocess: bool = True,
    min_confidence: float = 40.0,
) -> OcrResult:
    source = preprocess(image) if use_preprocess else image
    scale = source.shape[1] / image.shape[1]
    config = f"--psm {psm}"
    try:
        text = pytesseract.image_to_string(source, lang=lang, config=config)
        data = pytesseract.image_to_data(
            source, lang=lang, config=config, output_type=Output.DICT
        )
    except TesseractNotFoundError as error:
        raise OcrUnavailable(INSTALL_HINT) from error

    words: list[Word] = []
    for index, raw in enumerate(data["text"]):
        content = raw.strip()
        if not content:
            continue
        confidence = float(data["conf"][index])
        if confidence < min_confidence:
            continue
        box = (
            int(data["left"][index] / scale),
            int(data["top"][index] / scale),
            int(data["width"][index] / scale),
            int(data["height"][index] / scale),
        )
        words.append(Word(content, box, confidence, int(data["line_num"][index])))
    return OcrResult(text=text.strip(), lang=lang, words=words)


def annotate(image: np.ndarray, result: OcrResult, show_boxes: bool = True) -> np.ndarray:
    canvas = render.as_bgr(image)
    if show_boxes:
        for word in result.words:
            x, y, w, h = word.box
            color = BOX_COLOR if word.confidence >= 70 else LOW_CONF_COLOR
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    for word in result.words:
        x, y, _, h = word.box
        size = max(12, min(26, int(h * 0.75)))
        canvas = render.put_text(canvas, word.text, (x, y + h + 2), size, TEXT_COLOR)
    return canvas


def side_by_side(image: np.ndarray, result: OcrResult, panel_width: int = 0) -> np.ndarray:
    annotated = annotate(image, result)
    height = annotated.shape[0]
    panel_width = panel_width or max(360, annotated.shape[1] // 2)
    panel = np.full((height, panel_width, 3), PANEL_BG, dtype=np.uint8)

    header = [
        f"Язык: {LANGUAGES.get(result.lang, result.lang)}",
        f"Слов распознано: {len(result.words)}",
        f"Средняя уверенность: {result.mean_confidence:.1f}%",
        "",
        "Распознанный текст:",
    ]
    panel = render.put_lines(panel, header, (16, 14), 17, render.ACCENT, spacing=6)
    offset = 14 + len(header) * 23 + 8
    body = _wrap(result.lines(), panel_width - 32, 18)
    panel = render.put_lines(panel, body, (16, offset), 18, render.CAPTION_FG, spacing=5)
    return np.hstack([annotated, panel])


def _wrap(lines: list[str], width: int, size: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        current = ""
        for word in line.split():
            candidate = f"{current} {word}".strip()
            if render.text_size(candidate, size)[0] <= width or not current:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped
