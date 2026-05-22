"""Benchmark runner for EchoFace Lite pipeline validation."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from ecoface_lite.ai_engine.bootstrap import get_recognition_pipeline
from ecoface_lite.ai_engine.diagnostics import diagnostics
from ecoface_lite.core.config import get_settings
from ecoface_lite.core.metrics import metrics
from ecoface_lite.core.stability_scoring import StabilityScorer
from ecoface_lite.db.session import get_session_factory
from ecoface_lite.services.video_service import process_prerecorded_video

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK_SCENARIOS = [
    {"id": "dense_crowd_daylight", "name": "Dense Crowd Daylight CCTV", "video": "benchmarks/dense_crowd.mp4"},
    {"id": "low_light_station", "name": "Low-light Station Footage", "video": "benchmarks/low_light.mp4"},
    {"id": "indoor_frontal", "name": "Clean Face Frontal Indoor", "video": "benchmarks/indoor_frontal.mp4"},
    {"id": "side_profile_walking", "name": "Side-profile Walking Motion", "video": "benchmarks/side_profile.mp4"},
    {"id": "partial_occlusion", "name": "Partial Occlusion Crossing", "video": "benchmarks/occlusion.mp4"},
    {"id": "fast_motion", "name": "Fast Motion Crossing", "video": "benchmarks/fast_motion.mp4"},
    {"id": "mixed_scale_depth", "name": "Mixed-scale Crowd Depth", "video": "benchmarks/mixed_scale.mp4"},
]

async def run_benchmark(scenario: Dict[str, str], version: str = "v1"):
    """Run a single benchmark scenario and export results."""
    settings = get_settings()
    pipeline = get_recognition_pipeline()
    session_factory = get_session_factory()
    scorer = StabilityScorer()
    
    logger.info(f"Starting benchmark: {scenario['name']}")
    
    # Reset metrics for clean run
    metrics.reset()
    diagnostics.reset()
    
    mode = "hybrid" # Current validated baseline
    
    try:
        async with session_factory() as session:
            # We assume the videos exist in settings.resolved_videos_dir() / "benchmarks/"
            # For the experiment, if they don't exist, we skip with a warning.
            video_path = settings.resolved_videos_dir() / scenario["video"]
            if not video_path.exists():
                logger.warning(f"Benchmark video not found: {video_path}. Skipping.")
                return

            result = await process_prerecorded_video(
                session,
                pipeline,
                settings,
                video_relative_path=scenario["video"],
                job_id=f"bench_{scenario['id']}"
            )
            
            # Collect telemetry
            snapshot = metrics.snapshot()
            raw_metrics = {
                "fps": snapshot.rates.get("average_processing_fps", 0.0),
                "detector_runtime_ms": snapshot.averages.get("detector_runtime_ms", 0.0),
                "avg_track_duration": snapshot.averages.get("avg_track_duration", 0.0),
                "identity_switches": snapshot.counters.get("identity_switches", 0),
                "detector_over_budget_count": snapshot.counters.get("detector_over_budget_count", 0),
                "resolution_overflow_count": snapshot.counters.get("resolution_clamped_down_count", 0),
                "avg_bbox_delta_before": snapshot.averages.get("avg_bbox_delta_before", 0.0),
                "recovery_success_rate": snapshot.rates.get("recovery_success_rate", 0.0),
                "identity_temporal_confidence_avg": snapshot.averages.get("identity_temporal_confidence_avg", 0.0),
            }
            
            # Calculate Stability Score
            scores = scorer.calculate_scores(raw_metrics)
            
            # Prepare Export
            diag_snapshot = diagnostics.snapshot()
            regression_warnings = [
                e["reason"] for e in diag_snapshot.get("recent_events", [])
                if e["category"] == "regression"
            ]

            export_data = {
                "scenario": scenario["id"],
                "mode": mode,
                "version": version,
                "metrics": raw_metrics,
                "scores": {
                    "overall": scores.overall,
                    "tracking": scores.tracking,
                    "identity": scores.identity,
                    "detector": scores.detector,
                    "grade": scores.grade
                },
                "regression_warnings": regression_warnings
            }
            
            # Naming format: <scenario>_<mode>_<version>.json
            filename = f"{scenario['id']}_{mode}_hardened_{version}.json"
            export_path = settings.project_root / "data" / "benchmarks" / filename
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Benchmark completed: {scenario['id']}. Results at {export_path}")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {scenario['id']}. Error: {str(e)}")

async def main():
    version = "v1"
    for scenario in BENCHMARK_SCENARIOS:
        await run_benchmark(scenario, version)

if __name__ == "__main__":
    asyncio.run(main())
