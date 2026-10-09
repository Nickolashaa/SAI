from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .contours import Shape, build_shape, group_by_shape

BINARIZATIONS = ("otsu", "adaptive", "fixed")


@dataclass
class PipelineParams:
    binarization: str = "otsu"
    threshold: int = 127
    block_size: int = 31
    constant: int = 10
    kernel_size: int = 3
    open_iterations: int = 1
    close_iterations: int = 1
    min_area: int = 30
    max_area_ratio: float = 0.35
    min_side: int = 4
    equalize_points: int = 64
    group_threshold: float = 0.95
    external_only: bool = True


@dataclass
class PipelineResult:
    original: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    denoised: np.ndarray
    all_contours: list = field(default_factory=list)
    shapes: list[Shape] = field(default_factory=list)
    groups: int = 0

    @property
    def rejected(self) -> int:
        return len(self.all_contours) - len(self.shapes)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray, params: PipelineParams) -> np.ndarray:
    if params.binarization == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary
    if params.binarization == "adaptive":
        block = max(3, params.block_size | 1)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block, params.constant,
        )
    _, binary = cv2.threshold(gray, params.threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def denoise(binary: np.ndarray, params: PipelineParams) -> np.ndarray:
    size = max(1, params.kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    cleaned = binary
    if params.open_iterations:
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, kernel, iterations=params.open_iterations
        )
    if params.close_iterations:
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=params.close_iterations
        )
    return cleaned


def find_contours(binary: np.ndarray, params: PipelineParams) -> list:
    mode = cv2.RETR_EXTERNAL if params.external_only else cv2.RETR_LIST
    contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_NONE)
    return list(contours)


def filter_contours(contours: list, shape: tuple[int, int], params: PipelineParams) -> list:
    height, width = shape
    limit = height * width * params.max_area_ratio
    kept = []
    for contour in contours:
        if len(contour) < 5:
            continue
        area = cv2.contourArea(contour)
        if area < params.min_area or area > limit:
            continue
        _, _, w, h = cv2.boundingRect(contour)
        if w < params.min_side or h < params.min_side:
            continue
        kept.append(contour)
    return kept


def run(image: np.ndarray, params: PipelineParams) -> PipelineResult:
    gray = to_gray(image)
    binary = binarize(gray, params)
    cleaned = denoise(binary, params)
    contours = find_contours(cleaned, params)
    kept = filter_contours(contours, gray.shape[:2], params)
    shapes = [build_shape(i, c, params.equalize_points) for i, c in enumerate(kept)]
    groups = group_by_shape(shapes, params.group_threshold)
    return PipelineResult(
        original=image,
        gray=gray,
        binary=binary,
        denoised=cleaned,
        all_contours=contours,
        shapes=shapes,
        groups=groups,
    )
