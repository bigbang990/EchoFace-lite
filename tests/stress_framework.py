"""Stress testing framework for EchoFace Lite Adaptive Recall Preservation."""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from ecoface_lite.ai_engine.detector import BoundingBox, DetectedFace, FaceLandmarks
from ecoface_lite.ai_engine.pipeline import RecognitionPipeline
from ecoface_lite.core.config import Settings
from ecoface_lite.core.metrics import metrics

@dataclass
class SimulatedObject:
    id_name: str
    true_id: int
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    embedding_base: np.ndarray = field(default_factory=lambda: np.random.randn(512).astype(np.float32))
    noise_level: float = 0.01
    occlusion: float = 0.0 # 0 to 1
    det_score_base: float = 0.95
    is_active: bool = True

    def step(self):
        self.x += self.vx
        self.y += self.vy
        
    def get_detected_face(self, frame_index: int) -> Optional[DetectedFace]:
        if not self.is_active or self.occlusion > 0.8:
            return None
            
        # Add jitter
        jitter_x = np.random.normal(0, 1.0)
        jitter_y = np.random.normal(0, 1.0)
        
        # Simulated det score influenced by occlusion
        det_score = self.det_score_base - (self.occlusion * 0.5)
        
        # Fake landmarks to avoid mismatch rejections
        # points: left_eye, right_eye, nose, left_mouth, right_mouth
        cx = self.x + self.w / 2
        cy = self.y + self.h / 2
        landmarks = FaceLandmarks(points=np.array([
            [cx - 10, cy - 10], # LE
            [cx + 10, cy - 10], # RE
            [cx, cy],           # Nose
            [cx - 10, cy + 10], # LM
            [cx + 10, cy + 10]  # RM
        ], dtype=np.float32))

        return DetectedFace(
            bbox=BoundingBox(
                x1=self.x + jitter_x,
                y1=self.y + jitter_y,
                x2=self.x + self.w + jitter_x,
                y2=self.y + self.h + jitter_y
            ),
            det_score=float(det_score),
            landmarks=landmarks
        )

    def get_embedding(self) -> np.ndarray:
        # Simulate slight drift/noise in embeddings
        drift = np.random.normal(0, self.noise_level, 512).astype(np.float32)
        emb = self.embedding_base + drift
        return emb / np.linalg.norm(emb)

class MockEmbedder:
    def __init__(self, world: 'SimulationWorld'):
        self.world = world
    def embed_face(self, frame, face: DetectedFace):
        # Match detected face to world object based on IoU
        best_obj = None
        max_iou = 0.1
        for obj in self.world.objects:
            iou = self._iou(face.bbox, BoundingBox(obj.x, obj.y, obj.x+obj.w, obj.y+obj.h))
            if iou > max_iou:
                max_iou = iou
                best_obj = obj
        
        if best_obj:
            return best_obj.get_embedding()
        return np.random.randn(512).astype(np.float32)

    def _iou(self, b1, b2):
        ix1 = max(b1.x1, b2.x1)
        iy1 = max(b1.y1, b2.y1)
        ix2 = min(b1.x2, b2.x2)
        iy2 = min(b1.y2, b2.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1)
        area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1)
        return inter / (area1 + area2 - inter + 1e-6)

class MockDetector:
    def __init__(self, world: 'SimulationWorld'):
        self.world = world
    def detect(self, frame):
        # We need to return faces in the coordinate system of 'frame'
        # frame is (input_height, input_width, 3)
        # world.frame_bgr is (1080, 1920, 3)
        scale_x = frame.shape[1] / 1920.0
        scale_y = frame.shape[0] / 1080.0

        faces = []
        for obj in self.world.objects:
            face = obj.get_detected_face(self.world.frame_index)
            if face:
                # Scale face back to detection frame coordinates
                scaled_face = DetectedFace(
                    bbox=BoundingBox(
                        x1=face.bbox.x1 * scale_x,
                        y1=face.bbox.y1 * scale_y,
                        x2=face.bbox.x2 * scale_x,
                        y2=face.bbox.y2 * scale_y
                    ),
                    det_score=face.det_score,
                    landmarks=FaceLandmarks(points=face.landmarks.points * [scale_x, scale_y]) if face.landmarks else None
                )
                faces.append(scaled_face)
        return faces

class SimulationWorld:
    def __init__(self):
        self.objects: List[SimulatedObject] = []
        self.frame_index = 0
        # Initialize with noise to avoid 'too blurry' rejections
        # Use more high-frequency noise (checkerboard-like) to ensure Laplacian variance
        self.frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.frame_bgr[::2, ::2] = 255
        self.frame_bgr[1::2, 1::2] = 255

    def add_object(self, obj: SimulatedObject):
        self.objects.append(obj)

    def step(self):
        for obj in self.objects:
            obj.step()
        self.frame_index += 1

class WorldPrepared:
    def __init__(self, bgr):
        self.bgr = bgr
        # Add diagnostics mock that responds to .get()
        class MockDiag:
            def get(self, key, default=None): return default
        self.diagnostics = MockDiag()

class MockMatcher:
    def match(self, emb, gallery): return None
    def best_match(self, emb, gallery, threshold): return None
    def top_k(self, query, gallery, k): return []

def run_simulation(
    pipeline: RecognitionPipeline, 
    world: SimulationWorld, 
    steps: int,
    callback=None
):
    results = []
    
    for _ in range(steps):
        world.step()
        # Simulate time passing for duration_ms to work
        time.sleep(0.005) 
        
        # Use public API to ensure all stages (preprocess, governance, path decision) run
        matches = pipeline.process_frame(
            frame_bgr=world.frame_bgr,
            frame_index=world.frame_index,
            gallery=[]
        )
        
        if callback:
            callback(world.frame_index, pipeline, matches)
        results.append(matches)
    return results
