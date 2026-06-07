from pathlib import Path
import urllib.request, sys

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEST = WEIGHTS_DIR / "yolov8n-face.pt"
URL = (
    "https://huggingface.co/arnabdhar/"
    "YOLOv8-Face-Detection/resolve/main/model.pt"
)
MIN_SIZE = 5_000_000

def download():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    print(f"Downloading to {DEST} ...")
    urllib.request.urlretrieve(URL, DEST)
    size = DEST.stat().st_size
    if size < MIN_SIZE:
        DEST.unlink()
        print(f"FAILED: file too small ({size} bytes) — corrupt download")
        sys.exit(1)
    print(f"OK — {size / 1e6:.1f} MB saved to {DEST}")

if __name__ == "__main__":
    download()
