from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def to_complex(contour: np.ndarray) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float64)
    return points[:, 0] + 1j * points[:, 1]


def to_points(values: np.ndarray) -> np.ndarray:
    return np.stack([values.real, values.imag], axis=1).astype(np.int32)


def elementary_vectors(points: np.ndarray) -> np.ndarray:
    return np.roll(points, -1) - points


def resample(points: np.ndarray, k: int) -> np.ndarray:
    closed = np.append(points, points[:1])
    steps = np.abs(np.diff(closed))
    perimeter = steps.sum()
    if perimeter <= 0 or len(points) < 2:
        return np.repeat(points[:1], k)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    targets = np.linspace(0.0, perimeter, k, endpoint=False)
    real = np.interp(targets, distance, closed.real)
    imag = np.interp(targets, distance, closed.imag)
    return real + 1j * imag


def equalize(contour: np.ndarray, k: int) -> np.ndarray:
    return resample(to_complex(contour), k)


def norm(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.abs(vectors * vectors.conjugate()).sum()))


def scalar_product(a: np.ndarray, b: np.ndarray) -> complex:
    return complex((a * b.conjugate()).sum())


def nsp(a: np.ndarray, b: np.ndarray) -> complex:
    denominator = norm(a) * norm(b)
    if denominator == 0:
        return 0j
    return scalar_product(a, b) / denominator


def cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(a) * np.fft.fft(b).conjugate()
    return np.fft.ifft(spectrum)


def autocorrelation(a: np.ndarray) -> np.ndarray:
    return cross_correlation(a, a)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = norm(a) * norm(b)
    if denominator == 0:
        return 0.0
    return float(np.abs(cross_correlation(a, b)).max() / denominator)


@dataclass
class Shape:
    index: int
    contour: np.ndarray
    equalized: np.ndarray
    vectors: np.ndarray
    box: tuple[int, int, int, int]
    area: float
    group: int = -1

    @property
    def perimeter(self) -> float:
        return float(cv2.arcLength(self.contour, True))


def build_shape(index: int, contour: np.ndarray, k: int) -> Shape:
    equalized = equalize(contour, k)
    return Shape(
        index=index,
        contour=contour,
        equalized=equalized,
        vectors=elementary_vectors(equalized),
        box=tuple(int(v) for v in cv2.boundingRect(contour)),
        area=float(cv2.contourArea(contour)),
    )


def group_by_shape(shapes: list[Shape], threshold: float = 0.95) -> int:
    representatives: list[np.ndarray] = []
    for shape in shapes:
        for group, reference in enumerate(representatives):
            if similarity(shape.vectors, reference) >= threshold:
                shape.group = group
                break
        else:
            shape.group = len(representatives)
            representatives.append(shape.vectors)
    return len(representatives)
