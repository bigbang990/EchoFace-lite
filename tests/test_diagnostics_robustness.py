
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ecoface_lite.ai_engine.diagnostics import DiagnosticsRecorder

def test_diagnostics_robustness():
    recorder = DiagnosticsRecorder()
    
    print("Testing record(metric, count=1)...")
    recorder.record("test", "metric_with_count", count=1)
    
    print("Testing record(metric, value=1)...")
    recorder.record("test", "metric_with_value", value=1)
    
    print("Testing record(metric, value=1, category='tracking')...")
    recorder.record("tracking", "metric_with_cat_and_val", value=1)
    
    print("Testing unknown kwargs...")
    recorder.record("test", "unknown_kwargs", something_new="data", count=5)
    
    print("Testing malformed kwargs (should not crash)...")
    # In Python, you can't really pass truly "malformed" kwargs to **kwargs easily, 
    # but we can test if our try-except handles internal logic errors if they existed.
    recorder.record("test", "malformed", metadata="not_a_dict")
    
    print("Testing increment()...")
    recorder.increment("inc_test", count=10)
    
    print("Testing timing()...")
    recorder.timing("timing_test", "slow_op", duration_ms=123.45)
    
    print("Testing warning()...")
    recorder.warning("warning_test", "something_is_wrong")

    snapshot = recorder.snapshot()
    health = snapshot["telemetry_health"]
    
    print("\nTelemetry Health Snapshot:")
    print(f"Contract Version: {health['contract_version']}")
    print(f"Unknown Kwarg Count: {health['unknown_kwarg_count']}")
    print(f"Normalized Calls: {health['normalized_calls']}")
    print(f"Interface Mismatch Count: {health['interface_mismatch_count']}")
    
    assert health['normalized_calls'] >= 2 # count=1 and increment(count=10)
    assert health['unknown_kwarg_count'] >= 3 # value=1 (twice), something_new="data"
    
    print("\nSUCCESS: DiagnosticsRecorder is now robust and flexible.")

if __name__ == "__main__":
    test_diagnostics_robustness()
