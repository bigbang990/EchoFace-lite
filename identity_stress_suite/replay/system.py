from __future__ import annotations
import time
import json
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

class DeterministicReplay:
    """
    Deterministic replay system for identity stress testing.
    Replays metadata or simulated detections to ensure repeatable tracker behavior.
    """
    
    def __init__(self, scenario_name: str, data_path: Optional[Path] = None):
        self.scenario_name = scenario_name
        self.data_path = data_path
        self.frames: List[Dict[str, Any]] = []
        
        if data_path and data_path.exists():
            self._load_data()

    def _load_data(self):
        with open(self.data_path, 'r') as f:
            self.frames = json.load(f)

    def save_scenario(self, output_path: Path):
        with open(output_path, 'w') as f:
            json.dump(self.frames, f, indent=2)

    def add_frame(self, frame_index: int, detections: List[Dict[str, Any]]):
        self.frames.append({
            "frame_index": frame_index,
            "detections": detections,
            "timestamp": time.time()
        })

    def get_frame(self, frame_index: int) -> Optional[List[Dict[str, Any]]]:
        for frame in self.frames:
            if frame["frame_index"] == frame_index:
                return frame["detections"]
        return None

    def run_replay(self, pipeline: Any, metrics_engine: Any):
        """Run the replay through the pipeline and collect metrics."""
        for frame_data in self.frames:
            frame_idx = frame_data["frame_index"]
            detections = frame_data["detections"]
            # Simulated update (implemented in StressSuiteRunner)
            pass

class RealVideoReplay:
    """
    Replay engine for processing actual MP4 video files through the production pipeline.
    Ensures deterministic frame processing and timestamp calculation.
    """
    
    def __init__(self, scenario_name: str, video_path: Path):
        self.scenario_name = scenario_name
        self.video_path = video_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 25.0 # Fallback
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frame_iterator(self, skip_frames: int = 0) -> Iterator[tuple[int, float, np.ndarray]]:
        """Iterate through video frames with index and timestamp, optionally skipping frames."""
        frame_idx = 0
        while True:
            # Skip frames using CAP_PROP_POS_FRAMES if skipping more than 1
            if skip_frames > 0 and frame_idx > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp_ms = (frame_idx / self.fps) * 1000.0
            yield frame_idx, timestamp_ms, frame
            frame_idx += (skip_frames + 1)
        self.cap.release()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

class ScenarioGenerator:
    """Utility to generate synthetic stress scenarios."""
    
    @staticmethod
    def create_crossing_scenario(duration_frames: int = 100) -> DeterministicReplay:
        replay = DeterministicReplay("crossing_people")
        np.random.seed(42) # Reproducible noise
        for i in range(duration_frames):
            # Person 1: Moving Left to Right with noise
            noise1 = np.random.normal(0, 1.5, 2)
            p1_x = 100 + i * 5 + noise1[0]
            p1_y = 200 + noise1[1]
            
            # Person 2: Moving Right to Left with noise
            noise2 = np.random.normal(0, 1.5, 2)
            p2_x = 600 - i * 5 + noise2[0]
            p2_y = 200 + noise2[1]
            
            detections = [
                {"bbox": (p1_x, p1_y, p1_x + 50, p1_y + 50), "score": 0.95, "gt_id": 1},
                {"bbox": (p2_x, p2_y, p2_x + 50, p2_y + 50), "score": 0.95, "gt_id": 2}
            ]
            replay.add_frame(i, detections)
        return replay

    @staticmethod
    def create_occlusion_scenario(duration_frames: int = 100) -> DeterministicReplay:
        replay = DeterministicReplay("occlusion")
        np.random.seed(42)
        for i in range(duration_frames):
            # Person moving steadily with noise
            noise = np.random.normal(0, 1.2, 2)
            p_x = 100 + i * 5 + noise[0]
            p_y = 200 + noise[1]
            
            detections = []
            # Disappear between frame 40 and 60
            if not (40 <= i <= 60):
                detections.append({"bbox": (p_x, p_y, p_x + 50, p_y + 50), "score": 0.95, "gt_id": 1})
            
            replay.add_frame(i, detections)
        return replay

    @staticmethod
    def create_reentry_scenario(duration_frames: int = 150) -> DeterministicReplay:
        replay = DeterministicReplay("re_entry")
        for i in range(duration_frames):
            p_x = 100 + i * 5
            detections = []
            # Exits at frame 50, re-enters at frame 100
            if i < 50 or i > 100:
                detections.append({"bbox": (p_x, 200, p_x + 50, 250), "score": 0.95, "gt_id": 1})
            replay.add_frame(i, detections)
        return replay

    @staticmethod
    def create_crowd_scenario(num_people: int = 15, duration_frames: int = 100) -> DeterministicReplay:
        replay = DeterministicReplay("crowd_density")
        for i in range(duration_frames):
            detections = []
            for p in range(num_people):
                # Randomish movement
                base_x = (p * 50) % 600
                p_x = base_x + (i * (p % 3 + 1))
                detections.append({"bbox": (p_x, 200 + (p * 5), p_x + 30, 230 + (p * 5)), "score": 0.90, "gt_id": p})
            replay.add_frame(i, detections)
        return replay

    @staticmethod
    def create_small_faces_scenario(duration_frames: int = 100) -> DeterministicReplay:
        replay = DeterministicReplay("small_faces")
        for i in range(duration_frames):
            p_x = 100 + i * 2
            # Very small face (20x20)
            detections = [{"bbox": (p_x, 200, p_x + 20, 220), "score": 0.85, "gt_id": 1}]
            replay.add_frame(i, detections)
        return replay

    @staticmethod
    def create_profile_rotation_scenario(duration_frames: int = 100) -> DeterministicReplay:
        replay = DeterministicReplay("profile_rotation")
        for i in range(duration_frames):
            p_x = 100 + i * 3
            # Score drops in middle as profile rotates
            score = 0.95
            if 40 <= i <= 60:
                score = 0.55 # Difficult profile angle
            
            detections = [{"bbox": (p_x, 200, p_x + 50, 250), "score": score, "gt_id": 1}]
            replay.add_frame(i, detections)
        return replay

    @staticmethod
    def create_partial_visibility_scenario(duration_frames: int = 100) -> DeterministicReplay:
        replay = DeterministicReplay("partial_visibility")
        for i in range(duration_frames):
            p_x = 100 + i * 3
            # Low confidence due to partial visibility
            detections = [{"bbox": (p_x, 200, p_x + 50, 250), "score": 0.48, "gt_id": 1}]
            replay.add_frame(i, detections)
        return replay

    @staticmethod
    def create_same_clothes_scenario(duration_frames: int = 100) -> DeterministicReplay:
        replay = DeterministicReplay("same_clothes")
        for i in range(duration_frames):
            # Two people moving very close to each other
            p1_x = 100 + i * 5
            p2_x = 110 + i * 5 # Only 10 pixels apart
            
            detections = [
                {"bbox": (p1_x, 200, p1_x + 50, 250), "score": 0.95, "gt_id": 1},
                {"bbox": (p2_x, 200, p2_x + 50, 250), "score": 0.95, "gt_id": 2}
            ]
            replay.add_frame(i, detections)
        return replay
