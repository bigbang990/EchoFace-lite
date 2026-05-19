"""Core experiment exporter for EchoFace dashboard observability."""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ecoface_lite.core.config import Settings
from ecoface_lite.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for experiment export."""
    export_format: str = "zip"  # "zip", "json", "csv"
    include_screenshots: bool = True
    include_false_positives: bool = True
    include_graph_data: bool = True
    include_event_timeline: bool = True
    compress_images: bool = True


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment session."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    video_name: str = ""
    video_duration: float = 0.0
    frame_count: int = 0
    test_operator: str = ""
    pipeline_version: str = "2A.5"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "video_name": self.video_name,
            "video_duration": self.video_duration,
            "frame_count": self.frame_count,
            "test_operator": self.test_operator,
            "pipeline_version": self.pipeline_version,
            "notes": self.notes,
        }


class ExperimentExporter:
    """Export experiment session data for dashboard observability."""

    def __init__(
        self,
        settings: Settings,
        config: ExportConfig | None = None,
    ) -> None:
        self._settings = settings
        self._config = config or ExportConfig()
        self._metadata = ExperimentMetadata()
        self._event_timeline = None  # Will be set separately
        self._notes_tracker = None  # Will be set separately

    def set_metadata(
        self,
        video_name: str,
        video_duration: float = 0.0,
        frame_count: int = 0,
        test_operator: str = "",
        notes: str = "",
    ) -> None:
        """Set experiment metadata.

        Args:
            video_name: Name of the video file
            video_duration: Duration in seconds
            frame_count: Total number of frames
            test_operator: Name of the test operator
            notes: Experimental notes
        """
        self._metadata = ExperimentMetadata(
            video_name=video_name,
            video_duration=video_duration,
            frame_count=frame_count,
            test_operator=test_operator,
            notes=notes,
        )

    def export_session(
        self,
        output_dir: Path,
        metrics_data: dict[str, Any] | None = None,
    ) -> Path:
        """Export complete experiment session.

        Args:
            output_dir: Base directory for exports
            metrics_data: Detection metrics data

        Returns:
            Path to exported file or directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"experiment_{self._metadata.experiment_id[:8]}_{timestamp}"

        if self._config.export_format == "zip":
            return self._export_zip(output_dir, export_name, metrics_data)
        elif self._config.export_format == "json":
            return self._export_json(output_dir, export_name, metrics_data)
        elif self._config.export_format == "csv":
            return self._export_csv(output_dir, export_name, metrics_data)
        else:
            raise ValueError(f"Unsupported export format: {self._config.export_format}")

    def _export_zip(
        self,
        output_dir: Path,
        export_name: str,
        metrics_data: dict[str, Any] | None = None,
    ) -> Path:
        """Export as ZIP package.

        Args:
            output_dir: Base directory for exports
            export_name: Name of the export
            metrics_data: Detection metrics data

        Returns:
            Path to ZIP file
        """
        temp_dir = output_dir / f"temp_{export_name}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create subdirectories
            screenshots_dir = temp_dir / "screenshots"
            false_positives_dir = temp_dir / "false_positive_samples"
            missed_faces_dir = temp_dir / "missed_face_annotations"

            if self._config.include_screenshots:
                screenshots_dir.mkdir(exist_ok=True)
            if self._config.include_false_positives:
                false_positives_dir.mkdir(exist_ok=True)
            missed_faces_dir.mkdir(exist_ok=True)

            # Export individual files
            self._export_experiment_metadata(temp_dir)
            self._export_feature_flags(temp_dir)
            self._export_config_snapshot(temp_dir)
            self._export_metrics(temp_dir, metrics_data)
            self._export_graph_data(temp_dir, metrics_data)
            self._export_event_timeline(temp_dir)
            self._export_false_positives(temp_dir, false_positives_dir)
            self._export_system_info(temp_dir)
            self._export_notes(temp_dir)

            # Create ZIP
            zip_path = output_dir / f"{export_name}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)

            logger.info("Exported experiment to ZIP: %s", zip_path)

            # Cleanup temp directory
            shutil.rmtree(temp_dir)

            return zip_path

        except Exception as e:
            # Cleanup on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            logger.error("Failed to export ZIP: %s", e)
            raise

    def _export_json(
        self,
        output_dir: Path,
        export_name: str,
        metrics_data: dict[str, Any] | None = None,
    ) -> Path:
        """Export as single JSON file.

        Args:
            output_dir: Base directory for exports
            export_name: Name of the export
            metrics_data: Detection metrics data

        Returns:
            Path to JSON file
        """
        export_data = {
            "experiment": self._metadata.to_dict(),
            "feature_flags": self._collect_feature_flags(),
            "config": self._collect_config_snapshot(),
            "metrics": metrics_data or {},
            "system_info": self._collect_system_info(),
        }

        json_path = output_dir / f"{export_name}.json"
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info("Exported experiment to JSON: %s", json_path)
        return json_path

    def _export_csv(
        self,
        output_dir: Path,
        export_name: str,
        metrics_data: dict[str, Any] | None = None,
    ) -> Path:
        """Export metrics as CSV.

        Args:
            output_dir: Base directory for exports
            export_name: Name of the export
            metrics_data: Detection metrics data

        Returns:
            Path to CSV file
        """
        import csv

        csv_path = output_dir / f"{export_name}_metrics.csv"

        if metrics_data and "per_frame_metrics" in metrics_data:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_data["per_frame_metrics"][0].keys())
                writer.writeheader()
                writer.writerows(metrics_data["per_frame_metrics"])

        logger.info("Exported metrics to CSV: %s", csv_path)
        return csv_path

    def _export_experiment_metadata(self, export_dir: Path) -> None:
        """Export experiment metadata to experiment.json."""
        metadata_path = export_dir / "experiment.json"
        with open(metadata_path, 'w') as f:
            json.dump(self._metadata.to_dict(), f, indent=2)

    def _export_feature_flags(self, export_dir: Path) -> None:
        """Export feature flags to feature_flags.json."""
        flags_path = export_dir / "feature_flags.json"
        with open(flags_path, 'w') as f:
            json.dump(self._collect_feature_flags(), f, indent=2)

    def _export_config_snapshot(self, export_dir: Path) -> None:
        """Export configuration snapshot to config_snapshot.json."""
        config_path = export_dir / "config_snapshot.json"
        with open(config_path, 'w') as f:
            json.dump(self._collect_config_snapshot(), f, indent=2, default=str)

    def _export_metrics(
        self,
        export_dir: Path,
        metrics_data: dict[str, Any] | None = None,
    ) -> None:
        """Export metrics to metrics.json."""
        metrics_path = export_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data or {}, f, indent=2, default=str)

    def _export_graph_data(
        self,
        export_dir: Path,
        metrics_data: dict[str, Any] | None = None,
    ) -> None:
        """Export graph data to graphs.json."""
        if not self._config.include_graph_data or not metrics_data:
            return

        graph_path = export_dir / "graphs.json"
        graph_data = self._extract_graph_data(metrics_data)

        with open(graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)

    def _export_event_timeline(self, export_dir: Path) -> None:
        """Export event timeline to event_timeline.json."""
        if not self._config.include_event_timeline or not self._event_timeline:
            return

        timeline_path = export_dir / "event_timeline.json"
        with open(timeline_path, 'w') as f:
            json.dump(self._event_timeline.to_dict(), f, indent=2, default=str)

    def _export_false_positives(
        self,
        export_dir: Path,
        false_positives_dir: Path,
    ) -> None:
        """Export false positive samples."""
        if not self._config.include_false_positives:
            return

        # Copy false positive snapshots from their configured location
        fp_source = self._settings.resolved_false_positive_dataset_dir()
        if fp_source.exists():
            fp_metadata = export_dir / "false_positive_metadata.json"
            if (fp_source / "metadata.json").exists():
                shutil.copy(fp_source / "metadata.json", fp_metadata)

    def _export_system_info(self, export_dir: Path) -> None:
        """Export system information to system_info.json."""
        system_path = export_dir / "system_info.json"
        with open(system_path, 'w') as f:
            json.dump(self._collect_system_info(), f, indent=2, default=str)

    def _export_notes(self, export_dir: Path) -> None:
        """Export experiment notes to notes.txt."""
        notes_path = export_dir / "notes.txt"
        with open(notes_path, 'w') as f:
            f.write(f"Experiment ID: {self._metadata.experiment_id}\n")
            f.write(f"Timestamp: {self._metadata.timestamp}\n")
            f.write(f"Video: {self._metadata.video_name}\n")
            f.write(f"Operator: {self._metadata.test_operator}\n")
            f.write(f"\nNotes:\n{self._metadata.notes}\n")

            if self._notes_tracker:
                f.write("\nAdjustments:\n")
                for adjustment in self._notes_tracker.get_adjustments():
                    f.write(f"- {adjustment}\n")

    def _collect_feature_flags(self) -> dict[str, Any]:
        """Collect feature flag snapshot."""
        return {
            "enable_multiscale_detection": self._settings.enable_multiscale_detection,
            "enable_tile_detection": self._settings.enable_tile_detection,
            "enable_weak_detection_memory": self._settings.enable_weak_detection_memory,
            "enable_confidence_normalization": self._settings.enable_confidence_normalization,
            "detection_metrics_enabled": self._settings.detection_metrics_enabled,
            "enable_legacy_face_validation": self._settings.enable_legacy_face_validation,
        }

    def _collect_config_snapshot(self) -> dict[str, Any]:
        """Collect configuration snapshot."""
        return {
            "detector": "SCRFD",
            "confidence_threshold": self._settings.confidence_large_face_threshold,
            "nms_threshold": 0.60,  # Default from detector
            "scales": self._settings.multiscale_scales if hasattr(self._settings, 'multiscale_scales') else [1.0],
            "tile_size": self._settings.tile_size if hasattr(self._settings, 'tile_size') else 640,
            "tile_overlap": self._settings.tile_overlap if hasattr(self._settings, 'tile_overlap') else 0.20,
            "weak_memory_frames": self._settings.weak_memory_max_frames if hasattr(self._settings, 'weak_memory_max_frames') else 32,
            "fusion_wbf_iou_threshold": self._settings.fusion_wbf_iou_threshold if hasattr(self._settings, 'fusion_wbf_iou_threshold') else 0.5,
            "fusion_crowd_iou_threshold": self._settings.fusion_crowd_iou_threshold if hasattr(self._settings, 'fusion_crowd_iou_threshold') else 0.3,
        }

    def _extract_graph_data(
        self,
        metrics_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract graph data from metrics."""
        graph_data = {}

        if metrics_data and "per_frame_metrics" in metrics_data:
            per_frame = metrics_data["per_frame_metrics"]

            # Detections over time
            graph_data["detections_over_time"] = [
                {"frame": m["frame_id"], "count": m["faces_detected"]}
                for m in per_frame
            ]

            # FPS over time
            if "detection_latency_ms" in per_frame[0]:
                graph_data["fps_over_time"] = [
                    {"frame": m["frame_id"], "fps": 1000.0 / m.get("detection_latency_ms", 1)}
                    for m in per_frame
                ]

            # Face size distribution
            graph_data["face_size_distribution"] = [
                {"frame": m["frame_id"], "avg_size": m.get("avg_face_size", 0)}
                for m in per_frame
            ]

            # Validator rejections
            graph_data["validator_rejections"] = [
                {"frame": m["frame_id"], "rejections": m.get("validator_rejections", 0)}
                for m in per_frame
            ]

        return graph_data

    def _collect_system_info(self) -> dict[str, Any]:
        """Collect system environment information."""
        import platform
        import psutil

        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu": "Unknown",  # Will be enhanced with GPU detection
            "vram_gb": 0,
            "onnx_provider": "CPUExecutionProvider",
            "cuda_enabled": False,
            "tensorrt_enabled": False,
        }

    def set_event_timeline(self, timeline) -> None:
        """Set event timeline collector."""
        self._event_timeline = timeline

    def set_notes_tracker(self, tracker) -> None:
        """Set experiment notes tracker."""
        self._notes_tracker = tracker
