"""
EchoFace Colab startup — single-server setup.

Run this in a Colab cell:
    !python scripts/colab_start.py

Requires:
    pip install pyngrok
    ngrok authtoken <your-token>   (free account at ngrok.com)

All routes (engine + INC) are served from a single uvicorn process on :8000.
For the two-server variant (Phase B), see the roadmap in AI_CONTEXT/roadmap.md.
"""

import subprocess
import time
import sys

# ── 1. install pyngrok if missing ────────────────────────────────────────────
try:
    from pyngrok import ngrok, conf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "-q"])
    from pyngrok import ngrok, conf

# ── 2. set ngrok auth token ──────────────────────────────────────────────────
# Paste your ngrok authtoken here or set env var NGROK_AUTHTOKEN
import os
NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN", "")
if NGROK_TOKEN:
    ngrok.set_auth_token(NGROK_TOKEN)
else:
    print("⚠  NGROK_AUTHTOKEN not set. Free tunnels may be limited.")
    print("   Set it: os.environ['NGROK_AUTHTOKEN'] = 'your_token'")
    print()

# ── 3. start core engine (port 8000) ─────────────────────────────────────────
print("Starting EchoFace on :8000 ...")
engine_proc = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn",
        "ecoface_lite.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# give server time to boot
print("Waiting for server to boot ...")
time.sleep(5)

# ── 4. open ngrok tunnel ──────────────────────────────────────────────────────
print("Opening ngrok tunnel ...")
engine_tunnel = ngrok.connect(8000, "http")

ENGINE_URL = engine_tunnel.public_url.replace("http://", "https://") + "/api/v1"

# ── 5. print connection info ──────────────────────────────────────────────────
print()
print("=" * 60)
print("  EchoFace — running on Colab")
print("=" * 60)
print()
print(f"  ENGINE URL  →  {ENGINE_URL}")
print()
print("  In the frontend BackendPanel:")
print(f"    Engine backend  →  paste ENGINE URL above")
print(f"    INC Server      →  paste ENGINE URL above (same server)")
print()
print("  Health check:")
print(f"    {ENGINE_URL.replace('/api/v1','')}/api/v1/health")
print("=" * 60)
print()
print("  Press Ctrl+C or interrupt the cell to stop.")

# ── 6. keep alive ─────────────────────────────────────────────────────────────
try:
    ngrok.get_ngrok_process().proc.wait()
except KeyboardInterrupt:
    print("\nShutting down ...")
    engine_proc.terminate()
    ngrok.kill()
    print("Done.")
