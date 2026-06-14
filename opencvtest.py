"""
AndroidCameraSource smoke test
Run: python smoke_android.py
Phone must be on same WiFi with IP Webcam app running
"""
import sys
sys.path.insert(0, ".")

from ecoface_lite.input_sources.android_source import AndroidCameraSource

STREAM_URL = "http://192.168.1.21:8080/video"

print("=" * 50)
print("AndroidCameraSource Smoke Test")
print("=" * 50)

source = AndroidCameraSource(
    source_id="android-smoke-01",
    name="Phone Camera",
    stream_url=STREAM_URL,
    zone="Test Zone",
    location="Local Test"
)

# Test 1 — connect
print("\n[1] connect()")
connected = source.connect()
print(f"    Result: {'✅ Connected' if connected else '❌ Failed'}")
if not connected:
    print("    STOP — cannot continue without connection")
    sys.exit(1)

# Test 2 — health check while connected
print("\n[2] health_check()")
health = source.health_check()
print(f"    Status: {health.status}")
print(f"    Consecutive failures: {source._consecutive_failures}")

# Test 3 — get 5 frames
print("\n[3] get_frame() x5")
success = 0
for i in range(5):
    frame = source.get_frame()
    if frame:
        success += 1
        print(f"    Frame {i+1}: ✅ shape={frame.bgr.shape} idx={frame.index}")
    else:
        print(f"    Frame {i+1}: ❌ None returned")

# Test 4 — metadata
print("\n[4] get_metadata()")
meta = source.get_metadata()
print(f"    Name:        {meta.name}")
print(f"    Source type: {meta.source_type}")
print(f"    Stream URL:  {meta.stream_url}")
print(f"    FPS:         {meta.fps}")
print(f"    Resolution:  {meta.width}x{meta.height}")

# Test 5 — capability flags
print("\n[5] capability flags")
print(f"    supports_live:       {source.supports_live}")
print(f"    supports_historical: {source.supports_historical}")
print(f"    supports_ptz:        {source.supports_ptz}")

# Test 6 — historical raises NotImplementedError
print("\n[6] get_historical_stream() raises NotImplementedError")
try:
    gen = source.get_historical_stream(None, None)
    next(gen)
    print("    ❌ Should have raised NotImplementedError")
except NotImplementedError:
    print("    ✅ Correctly raised NotImplementedError")
except Exception as e:
    print(f"    ❌ Wrong exception: {type(e).__name__}: {e}")

# Test 7 — disconnect
print("\n[7] disconnect()")
source.disconnect()
print(f"    Cap released: {'✅' if source._cap is None else '❌ still open'}")

# Test 8 — health after disconnect
print("\n[8] health_check() after disconnect")
health = source.health_check()
print(f"    Status: {health.status}  (expected: OFFLINE)")

print("\n" + "=" * 50)
print(f"Frames: {success}/5 captured")
print("✅ ALL TESTS PASSED" if success == 5 else f"⚠️  {5-success} frames failed")
print("=" * 50)