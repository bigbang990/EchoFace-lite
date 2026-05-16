from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ecoface_lite.core.config import Settings


@dataclass(frozen=True)
class FrameDiagnostics:
    brightness: float
    width: int
    height: int


@dataclass(frozen=True)
class PreprocessedFrame:
    bgr: np.ndarray
    diagnostics: FrameDiagnostics


class FramePreprocessor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def process(self, frame_bgr: np.ndarray) -> PreprocessedFrame:
        frame = self._resize(frame_bgr)
        brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        if self._settings.preprocessing_enable_clahe:
            frame = self._clahe(frame)
        if self._settings.preprocessing_enable_gamma and brightness < self._settings.preprocessing_brightness_target:
            frame = self._gamma(frame, self._settings.preprocessing_gamma)
        if self._settings.preprocessing_enable_denoise:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 4, 4, 7, 21)
        diagnostics = FrameDiagnostics(brightness=brightness, width=int(frame.shape[1]), height=int(frame.shape[0]))
        return PreprocessedFrame(bgr=frame, diagnostics=diagnostics)

    def _resize(self, frame_bgr: np.ndarray) -> np.ndarray:
        target_width = self._settings.preprocessing_max_width
        if target_width <= 0 or frame_bgr.shape[1] <= target_width:
            return frame_bgr
        ratio = target_width / frame_bgr.shape[1]
        height = max(1, int(frame_bgr.shape[0] * ratio))
        return cv2.resize(frame_bgr, (target_width, height), interpolation=cv2.INTER_AREA)

    def _clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(l_channel)
        merged = cv2.merge((enhanced, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _gamma(self, frame_bgr: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / max(gamma, 0.01)
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(frame_bgr, table)
