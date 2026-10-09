from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DETECTORS = ("sift", "akaze", "orb", "brisk")
BINARY_DETECTORS = {"akaze", "orb", "brisk"}


@dataclass
class Template:
    image: np.ndarray
    keypoints: tuple
    descriptors: np.ndarray | None
    name: str = "template"

    @property
    def corners(self) -> np.ndarray:
        h, w = self.image.shape[:2]
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


@dataclass
class MatchResult:
    keypoints: tuple = ()
    good: list = field(default_factory=list)
    inlier_flags: list[bool] = field(default_factory=list)
    homography: np.ndarray | None = None
    quad: np.ndarray | None = None

    @property
    def inliers(self) -> int:
        return sum(self.inlier_flags)

    @property
    def found(self) -> bool:
        return self.quad is not None


def _is_plausible(quad: np.ndarray, scene_shape: tuple[int, int]) -> bool:
    points = quad.reshape(-1, 2)
    if not np.all(np.isfinite(points)):
        return False
    if not cv2.isContourConvex(points.astype(np.float32)):
        return False
    area = abs(cv2.contourArea(points.astype(np.float32)))
    scene_area = scene_shape[0] * scene_shape[1]
    if area < scene_area * 0.002 or area > scene_area * 4.0:
        return False
    sides = [np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)]
    return min(sides) > 1e-3 and max(sides) / min(sides) < 12.0


class ObjectMatcher:
    def __init__(
        self,
        kind: str = "sift",
        ratio: float = 0.75,
        min_inliers: int = 10,
        max_features: int = 1500,
        reproj_threshold: float = 4.0,
    ) -> None:
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.max_features = max_features
        self.reproj_threshold = reproj_threshold
        self.template: Template | None = None
        self._set_kind(kind)

    def _set_kind(self, kind: str) -> None:
        if kind not in DETECTORS:
            raise ValueError(f"Неизвестный детектор: {kind}. Доступны: {', '.join(DETECTORS)}")
        self.kind = kind
        self.detector = self._create_detector(kind)
        self.binary = kind in BINARY_DETECTORS
        norm = cv2.NORM_HAMMING if self.binary else cv2.NORM_L2
        self.matcher = cv2.BFMatcher(norm)

    def _create_detector(self, kind: str):
        if kind == "sift":
            return cv2.SIFT_create(nfeatures=self.max_features)
        if kind == "orb":
            return cv2.ORB_create(nfeatures=self.max_features)
        if kind == "akaze":
            return cv2.AKAZE_create()
        return cv2.BRISK_create()

    def cycle_detector(self) -> str:
        nxt = DETECTORS[(DETECTORS.index(self.kind) + 1) % len(DETECTORS)]
        image = self.template.image if self.template else None
        name = self.template.name if self.template else "template"
        self._set_kind(nxt)
        if image is not None:
            self.set_template(image, name)
        return nxt

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def set_template(self, image: np.ndarray, name: str = "template") -> Template:
        keypoints, descriptors = self.detector.detectAndCompute(self.to_gray(image), None)
        if descriptors is None or len(keypoints) < 4:
            raise ValueError(
                "На изображении-образце найдено слишком мало ключевых точек. "
                "Возьмите более контрастный и детализированный объект."
            )
        self.template = Template(image, keypoints, descriptors, name)
        return self.template

    def load_template(self, path: Path) -> Template:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение-образец: {path}")
        return self.set_template(image, path.name)

    def _ratio_test(self, pairs) -> list:
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < self.ratio * second.distance:
                good.append(best)
        return good

    def match(self, scene: np.ndarray) -> MatchResult:
        if self.template is None:
            raise RuntimeError("Образец не задан")
        keypoints, descriptors = self.detector.detectAndCompute(self.to_gray(scene), None)
        if descriptors is None or len(keypoints) < 2:
            return MatchResult(keypoints=keypoints or ())

        pairs = self.matcher.knnMatch(self.template.descriptors, descriptors, k=2)
        good = self._ratio_test(pairs)
        result = MatchResult(keypoints=keypoints, good=good)
        if len(good) < max(4, self.min_inliers):
            return result

        src = np.float32([self.template.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.reproj_threshold)
        if homography is None:
            return result

        result.inlier_flags = [bool(v) for v in mask.ravel()]
        if result.inliers < self.min_inliers:
            return result

        quad = cv2.perspectiveTransform(self.template.corners, homography)
        if not _is_plausible(quad, scene.shape[:2]):
            return result

        result.homography = homography
        result.quad = quad
        return result
