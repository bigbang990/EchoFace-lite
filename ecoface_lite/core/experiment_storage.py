"""Versioned Experiment Storage Layer.

Stage 1 (Module M1): File system snapshot persistence under data/experiments/
with schema versioning. Retains all snapshots, UI loads latest 20 only.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Current schema version for experiment snapshots
SCHEMA_VERSION = 1


class BackendType(str, Enum):
    """Backend execution environment for experiment tagging."""
    
    LOCAL_CPU = "LOCAL_CPU"
    COLAB_GPU = "COLAB_GPU"
    REMOTE_GPU = "REMOTE_GPU"


class PipelineMode(str, Enum):
    """Pipeline execution mode for experiment tagging."""
    
    LEGACY_ONLY = "LEGACY_ONLY"
    HYBRID = "HYBRID"
    UNIFIED_ONLY = "UNIFIED_ONLY"


class RunStatus(str, Enum):
    """Execution status for experiment snapshots."""
    
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class ExperimentSnapshot:
    """Versioned experiment run snapshot with raw metrics only."""
    
    schema_version: int = SCHEMA_VERSION
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Session grouping for comparison UI
    experiment_session_id: str = ""
    
    # Environment metadata
    backend_type: BackendType = BackendType.LOCAL_CPU
    pipeline_mode: PipelineMode = PipelineMode.HYBRID
    
    # Execution status
    run_status: RunStatus = RunStatus.COMPLETED
    
    # Runtime configuration overrides
    runtime_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Raw metrics (direct storage, no derived abstractions)
    fps: float = 0.0
    detections: int = 0
    false_positives: int = 0
    validator_rejections: int = 0
    stable_matches: int = 0
    identity_switches: int = 0
    embedding_skips: int = 0
    track_fragmentation: float = 0.0
    avg_track_lifetime: float = 0.0
    detector_overload_events: int = 0
    
    # Additional execution metadata
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "experiment_session_id": self.experiment_session_id,
            "backend_type": self.backend_type.value,
            "pipeline_mode": self.pipeline_mode.value,
            "run_status": self.run_status.value,
            "runtime_overrides": self.runtime_overrides,
            "fps": self.fps,
            "detections": self.detections,
            "false_positives": self.false_positives,
            "validator_rejections": self.validator_rejections,
            "stable_matches": self.stable_matches,
            "identity_switches": self.identity_switches,
            "embedding_skips": self.embedding_skips,
            "track_fragmentation": self.track_fragmentation,
            "avg_track_lifetime": self.avg_track_lifetime,
            "detector_overload_events": self.detector_overload_events,
            "execution_metadata": self.execution_metadata,
        }


class ExperimentStorage:
    """Versioned experiment storage with full retention.
    
    Retains all snapshots indefinitely. UI loads latest 20 only.
    No automatic deletion - storage is cheap, debug history is priceless.
    """
    
    def __init__(self, storage_dir: Path):
        """Initialize experiment storage.
        
        Args:
            storage_dir: Base directory for experiment snapshots (data/experiments/).
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(
        self,
        experiment_session_id: str,
        backend_type: BackendType,
        pipeline_mode: PipelineMode,
        runtime_overrides: Dict[str, Any],
        fps: float,
        detections: int,
        false_positives: int = 0,
        validator_rejections: int = 0,
        stable_matches: int = 0,
        identity_switches: int = 0,
        embedding_skips: int = 0,
        track_fragmentation: float = 0.0,
        avg_track_lifetime: float = 0.0,
        detector_overload_events: int = 0,
        run_status: RunStatus = RunStatus.COMPLETED,
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentSnapshot:
        """Save experiment snapshot with raw metrics and versioning.
        
        Args:
            experiment_session_id: Session ID for grouping related experiments.
            backend_type: Backend execution environment.
            pipeline_mode: Pipeline execution mode.
            runtime_overrides: Runtime configuration overrides applied.
            fps: Frames per second achieved.
            detections: Total face detections.
            false_positives: False positive count.
            validator_rejections: Validator rejection count.
            stable_matches: Stable track matches.
            identity_switches: Identity switch events.
            embedding_skips: Embedding computation skips.
            track_fragmentation: Track fragmentation rate.
            avg_track_lifetime: Average track lifetime in frames.
            detector_overload_events: Detector overload event count.
            run_status: Execution status (COMPLETED, INTERRUPTED, FAILED, ABORTED).
            execution_metadata: Additional execution context metadata.
            
        Returns:
            Saved experiment snapshot instance.
        """
        # Create snapshot with mandatory schema version
        snapshot = ExperimentSnapshot(
            schema_version=SCHEMA_VERSION,
            experiment_session_id=experiment_session_id,
            backend_type=backend_type,
            pipeline_mode=pipeline_mode,
            run_status=run_status,
            runtime_overrides=runtime_overrides,
            fps=fps,
            detections=detections,
            false_positives=false_positives,
            validator_rejections=validator_rejections,
            stable_matches=stable_matches,
            identity_switches=identity_switches,
            embedding_skips=embedding_skips,
            track_fragmentation=track_fragmentation,
            avg_track_lifetime=avg_track_lifetime,
            detector_overload_events=detector_overload_events,
            execution_metadata=execution_metadata or {},
        )
        
        # Persist to file system
        self._persist_snapshot(snapshot)
        
        logger.info(f"Saved experiment snapshot: {snapshot.snapshot_id}")
        return snapshot
    
    def _persist_snapshot(self, snapshot: ExperimentSnapshot) -> None:
        """Persist snapshot to file system as JSON."""
        timestamp = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{timestamp}_{snapshot.snapshot_id[:8]}.json"
        filepath = self.storage_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot.to_dict(), f, indent=2)
            logger.debug(f"Persisted snapshot to {filepath}")
        except Exception as e:
            logger.error(f"Failed to persist snapshot {snapshot.snapshot_id}: {e}")
            raise
    
    def load_recent_snapshots(self, limit: Optional[int] = 20) -> List[Dict[str, Any]]:
        """Load recent snapshots from storage (UI-facing method).
        
        Args:
            limit: Maximum number of recent snapshots to return (default 20).
                   None returns all snapshots.
            
        Returns:
            List of snapshot dictionaries sorted by timestamp (newest first).
        """
        snapshot_files = sorted(self.storage_dir.glob("experiment_*.json"), reverse=True)
        
        # Fixed: Proper None handling without type ignore
        if limit is not None:
            snapshot_files = snapshot_files[:limit]
        
        snapshots = []
        
        for filepath in snapshot_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Fixed: Schema version check
                if data.get("schema_version") != SCHEMA_VERSION:
                    logger.warning(
                        f"Snapshot {filepath} has schema version {data.get('schema_version')}, "
                        f"expected {SCHEMA_VERSION}. May fail to load."
                    )
                    # Still attempt load, but warn
                
                snapshots.append(data)
            except Exception as e:
                logger.warning(f"Failed to load snapshot {filepath}: {e}")
                continue
        
        return snapshots
    
    def load_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the most recent snapshot from storage.
        
        Returns:
            Latest snapshot dictionary or None if storage is empty.
        """
        snapshots = self.load_recent_snapshots(limit=1)
        return snapshots[0] if snapshots else None
    
    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        """Load ALL snapshots from storage (admin/debug method).
        
        Returns:
            List of all snapshot dictionaries sorted by timestamp (newest first).
        """
        return self.load_recent_snapshots(limit=None)  # type: ignore
    
    def load_snapshots_by_session(self, experiment_session_id: str) -> List[Dict[str, Any]]:
        """Load all snapshots for a specific experiment session.
        
        Args:
            experiment_session_id: Session ID to filter by.
            
        Returns:
            List of snapshot dictionaries for the session sorted by timestamp (newest first).
        """
        all_snapshots = self.get_all_snapshots()
        return [
            s for s in all_snapshots 
            if s.get("experiment_session_id") == experiment_session_id
        ]
