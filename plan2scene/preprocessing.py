from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .config import DEFAULTS


@dataclass
class PreprocessResult:
    rgb_image: np.ndarray
    gray_image: np.ndarray
    binarized_image: np.ndarray
    edges_image: np.ndarray


def convert_pil_to_numpy(image) -> np.ndarray:
    return np.array(image)[:, :, ::-1]


def to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)


def denoise_image(gray_image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray_image, h=10)


def adaptive_binarize(gray_image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray_image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=2,
    )


def detect_edges(binary_image: np.ndarray) -> np.ndarray:
    return cv2.Canny(
        binary_image,
        threshold1=DEFAULTS.canny_threshold1,
        threshold2=DEFAULTS.canny_threshold2,
        L2gradient=True,
    )


def preprocess_pil_image(pil_image) -> PreprocessResult:
    bgr_image = convert_pil_to_numpy(pil_image)
    gray = to_grayscale(bgr_image)
    denoised = denoise_image(gray)
    binary = adaptive_binarize(denoised)
    edges = detect_edges(binary)
    return PreprocessResult(
        rgb_image=bgr_image,
        gray_image=gray,
        binarized_image=binary,
        edges_image=edges,
    )


def resize_to_max(image: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    max_w, max_h = max_size
    scale = min(max_w / width, max_h / height)
    if scale >= 1.0:
        return image
    new_w, new_h = int(width * scale), int(height * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
