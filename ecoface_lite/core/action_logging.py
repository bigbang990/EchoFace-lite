"""Non-Blocking Structured Action Logger.

Stage 1 (Module M1): Simple append-only JSONL logging with rotating files.
Thread-safe lock only, no background threads or daemons.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)


class DashboardAction(str, Enum):
    """8 core enum events for dashboard action tracking."""
    
    CONFIG_CHANGED = "CONFIG_CHANGED"
    MODE_CHANGED = "MODE_CHANGED"
    RESET_TRIGGERED = "RESET_TRIGGERED"
    CPU_PROTECTION_TRIGGERED = "CPU_PROTECTION_TRIGGERED"
    BACKEND_CONNECTED = "BACKEND_CONNECTED"
    VIDEO_PROCESS_STARTED = "VIDEO_PROCESS_STARTED"
    VIDEO_PROCESS_COMPLETED = "VIDEO_PROCESS_COMPLETED"
    EXPERIMENT_SNAPSHOT_SAVED = "EXPERIMENT_SNAPSHOT_SAVED"


ActorType = Literal["user", "system"]


@dataclass
class DashboardActionRecord:
    """Structured action log record with actor field."""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    actor: ActorType = "user"  # "user" or "system"
    event_type: DashboardAction = field(default=DashboardAction.CONFIG_CHANGED)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "event_type": self.event_type.value,
            "metadata": self.metadata,
        }


class DashboardActionLog:
    """Simple append-only JSONL action logger with rotating files.
    
    Thread-safe lock for concurrent writes. No background threads or daemons.
    """
    
    def __init__(
        self,
        log_dir: Path,
        max_file_size_mb: float = 10.0,
        max_log_files: int = 20,
    ):
        """Initialize action logger with rotating file configuration.
        
        Args:
            log_dir: Directory for persistent log files.
            max_file_size_mb: Maximum file size before rotation (in MB).
            max_log_files: Maximum number of log files to retain (cleanup oldest).
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.max_log_files = max_log_files
        self._lock = threading.Lock()
        
        self._current_log_file = self._get_current_log_file()
    
    def _get_current_log_file(self) -> Path:
        """Get current log file, rotate if size limit exceeded."""
        log_files = sorted(self.log_dir.glob("actions_*.jsonl"))
        
        # Check if latest file exists and is under size limit
        if log_files:
            latest = log_files[-1]
            # Fixed: check if file exists before .stat()
            if latest.exists() and latest.stat().st_size < self.max_file_size_bytes:
                return latest
        
        # Create new log file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"actions_{timestamp}.jsonl"
    
    def _cleanup_old_logs(self) -> None:
        """Remove oldest log files if retention limit exceeded."""
        log_files = sorted(self.log_dir.glob("actions_*.jsonl"))
        
        if len(log_files) > self.max_log_files:
            # Remove oldest files
            files_to_remove = log_files[:-self.max_log_files]
            for old_file in files_to_remove:
                try:
                    old_file.unlink()
                    logger.debug(f"Cleaned up old log file: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup old log file {old_file}: {e}")
    
    def _validate_actor(self, actor: str) -> ActorType:
        """Validate actor field to prevent invalid values."""
        if actor not in ("user", "system"):
            logger.warning(f"Invalid actor value '{actor}', defaulting to 'user'")
            return "user"
        return actor  # type: ignore
    
    def log_action(
        self,
        event_type: DashboardAction,
        actor: ActorType = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log action to JSONL file (thread-safe).
        
        Args:
            event_type: The dashboard action enum value.
            actor: "user" or "system" (for automated events like CPU protection).
            metadata: Optional metadata dictionary for the action.
        """
        # Validate actor field
        validated_actor = self._validate_actor(actor)
        
        record = DashboardActionRecord(
            event_type=event_type,
            actor=validated_actor,
            metadata=metadata or {},
        )
        
        with self._lock:
            # Check rotation
            if self._current_log_file.exists() and self._current_log_file.stat().st_size >= self.max_file_size_bytes:
                self._current_log_file = self._get_current_log_file()
                self._cleanup_old_logs()
            
            # Append to JSONL
            try:
                with open(self._current_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
            except Exception as e:
                logger.error(f"Failed to write action log: {e}")
        
        logger.debug(f"Logged action: {event_type.value} (actor: {validated_actor})")
    
    def get_recent_actions(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Get recent actions from log files.
        
        Args:
            limit: Maximum number of recent actions to retrieve.
            
        Returns:
            List of action dictionaries sorted by timestamp (newest first).
        """
        actions = []
        log_files = sorted(self.log_dir.glob("actions_*.jsonl"), reverse=True)
        
        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(actions) >= limit:
                            break
                        try:
                            actions.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
                continue
            
            if len(actions) >= limit:
                break
        
        return actions[:limit]
    
    def get_action_count(self, event_type: DashboardAction) -> int:
        """Count occurrences of specific action in current log file.
        
        Args:
            event_type: The dashboard action enum to count.
            
        Returns:
            Count of actions in current log file matching the specified type.
        """
        count = 0
        
        try:
            if self._current_log_file.exists():
                with open(self._current_log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            if record.get("event_type") == event_type.value:
                                count += 1
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Failed to count actions: {e}")
        
        return count
