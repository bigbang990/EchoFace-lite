import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import deque

class VisualOverlayEngine:
    """
    Visualization layer for identity continuity debugging.
    Handles persistent coloring, motion trails, and state display.
    """
    
    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        self.track_colors: Dict[str, Tuple[int, int, int]] = {}
        self.track_history: Dict[str, deque] = {} # track_id -> deque of centroids
        
    def get_color(self, track_id: str) -> Tuple[int, int, int]:
        """Deterministic color mapping for track IDs."""
        if track_id not in self.track_colors:
            # Generate deterministic color based on hash of track_id
            h = hash(track_id)
            r = (h & 0xFF0000) >> 16
            g = (h & 0x00FF00) >> 8
            b = h & 0x0000FF
            # Ensure it's not too dark
            r = (r % 200) + 55
            g = (g % 200) + 55
            b = (b % 200) + 55
            self.track_colors[track_id] = (b, g, r) # OpenCV uses BGR
        return self.track_colors[track_id]

    def draw_overlay(self, frame: np.ndarray, active_tracks: List[Dict[str, Any]]) -> np.ndarray:
        overlay = frame.copy()
        
        for track in active_tracks:
            tid = track["track_id"]
            bbox = track["bbox"] # (x1, y1, x2, y2)
            conf = track.get("confidence", 0.0)
            age = track.get("visibility_age", 0)
            
            # ── 1. Advanced State Mapping ─────────────────────────────────────
            # Map tracker states to visual debug states
            raw_state = track.get("state", "VISIBLE").upper()
            lost_frames = track.get("lost_frames", 0)
            
            if lost_frames > 0:
                visual_state = "OCCLUDED"
            elif raw_state == "STABLE" and track.get("identity") is not None:
                visual_state = "REIDENTIFIED"
            elif track.get("recovery_count", 0) > 0 and age < 10: # Recently recovered
                visual_state = "RECOVERING"
            else:
                visual_state = "VISIBLE"
            
            color = self.get_color(tid)
            x1, y1, x2, y2 = map(int, bbox)
            
            # ── 2. Confidence Heat Display ────────────────────────────────────
            # High confidence (1.0) -> Green/Stable, Low confidence (0.0) -> Red/Warning
            # In BGR: Green is (0, 255, 0), Red is (0, 0, 255)
            heat_b = int(color[0] * conf)
            heat_g = int(255 * conf)
            heat_r = int(255 * (1.0 - conf))
            heat_color = (heat_b, heat_g, heat_r)
            
            # 3. Persistent Bounding Box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), heat_color, 2)
            
            # Label background for readability
            label = f"ID:{tid} CONF:{conf:.2f} AGE:{age}f"
            cv2.rectangle(overlay, (x1, y1 - 35), (x1 + 220, y1), (0, 0, 0), -1)
            cv2.putText(overlay, label, (x1 + 5, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # 4. State Marker
            state_label = f"STATE: {visual_state}"
            cv2.putText(overlay, state_label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, heat_color, 1)
            
            # 5. Motion Trails
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=self.history_len)
            self.track_history[tid].append((cx, cy))
            
            history = list(self.track_history[tid])
            for i in range(1, len(history)):
                # Fading effect: older points are thinner and less opaque (simulated via thickness)
                thickness = max(1, int((i / len(history)) * 3))
                pt1 = history[i-1]
                pt2 = history[i]
                cv2.line(overlay, pt1, pt2, heat_color, thickness)

        return overlay

    def export_video(self, frames: List[np.ndarray], output_path: str, fps: int = 20):
        if not frames:
            return
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
