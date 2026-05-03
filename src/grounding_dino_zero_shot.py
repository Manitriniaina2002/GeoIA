"""Grounding DINO zero-shot stadium detection helpers.

This module ports the tiled, prompt-driven zero-shot pipeline into GeoIA.
It keeps the output in pixel-space so the existing geospatial wrappers can
convert boxes to polygons in QGIS or overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


DEFAULT_STADIUM_PROMPT = (
    "stadium. sports field. arena. athletics track. soccer stadium. football stadium."
)


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    finite = array[np.isfinite(array)] if np.isfinite(array).any() else None
    if finite is not None and finite.size:
        p2, p98 = np.percentile(finite, (2, 98))
    else:
        p2 = float(np.min(array))
        p98 = float(np.max(array))
    if p98 > p2:
        array = np.clip(array, p2, p98)
        array = ((array - p2) / (p98 - p2) * 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


@dataclass
class DetectionConfig:
    text_prompt: str = DEFAULT_STADIUM_PROMPT
    tile_size: int = 800
    tile_overlap: int = 100
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: Optional[str] = None


class RasterTiler:
    def __init__(self, raster_path: str, tile_size: int = 800, overlap: int = 100):
        self.raster_path = raster_path
        self.tile_size = tile_size
        self.overlap = overlap

        with rasterio.open(raster_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.transform = src.transform
            self.bounds = src.bounds

    def get_tiles(self) -> List[Window]:
        tiles: List[Window] = []
        step = max(1, self.tile_size - self.overlap)
        for row_off in range(0, self.height, step):
            for col_off in range(0, self.width, step):
                win_width = min(self.tile_size, self.width - col_off)
                win_height = min(self.tile_size, self.height - row_off)
                if win_width < 100 or win_height < 100:
                    continue
                tiles.append(Window(col_off, row_off, win_width, win_height))
        return tiles

    def read_tile(self, window: Window) -> np.ndarray:
        with rasterio.open(self.raster_path) as src:
            if src.count >= 3:
                tile = src.read([1, 2, 3], window=window)
            else:
                tile = src.read(1, window=window)
                tile = np.stack([tile, tile, tile])

        tile = np.moveaxis(tile, 0, -1)
        if tile.dtype != np.uint8:
            tile = _normalize_to_uint8(tile)
        return tile


class GroundingDINORasterDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = config.device or self._default_device()
        self.processor = None
        self.model = None
        self._load_model()

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO requires torch and transformers. Install the optional ML dependencies first."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.config.model_id)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _detect_tile(self, tile: np.ndarray) -> List[Dict]:
        import torch

        pil_image = Image.fromarray(tile)
        inputs = self.processor(images=pil_image, text=self.config.text_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels")

        if boxes is None:
            return []

        boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        if labels is None:
            labels = ["stadium"] * len(boxes_np)

        detections: List[Dict] = []
        for box, score, label in zip(boxes_np, scores_np, labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "bbox": (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))),
                    "area": float((x2 - x1) * (y2 - y1)),
                    "score": float(score),
                    "label": str(label),
                }
            )
        return detections

    def detect_raster(self, raster_path: str, min_area: int = 5000) -> List[Dict]:
        tiler = RasterTiler(raster_path, self.config.tile_size, self.config.tile_overlap)
        detections: List[Dict] = []

        for window in tiler.get_tiles():
            tile = tiler.read_tile(window)
            tile_detections = self._detect_tile(tile)
            for det in tile_detections:
                x, y, w, h = det["bbox"]
                full_bbox = (int(window.col_off + x), int(window.row_off + y), int(w), int(h))
                area = float(w * h)
                if area < float(min_area):
                    continue
                detections.append(
                    {
                        "bbox": full_bbox,
                        "area": area,
                        "score": det["score"],
                        "label": det["label"],
                        "backend": "grounding_dino",
                        "prompt": self.config.text_prompt,
                    }
                )

        detections.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return detections


def detect_stadium_grounding_dino(
    raster_path: str,
    min_area: int = 5000,
    text_prompt: str = DEFAULT_STADIUM_PROMPT,
    tile_size: int = 800,
    tile_overlap: int = 100,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: Optional[str] = None,
) -> List[Dict]:
    config = DetectionConfig(
        text_prompt=text_prompt,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        model_id=model_id,
        device=device,
    )
    detector = GroundingDINORasterDetector(config)
    return detector.detect_raster(raster_path, min_area=min_area)