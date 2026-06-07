from pathlib import Path
import sys

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEST        = WEIGHTS_DIR / "yolov8n-face.pt"
MIN_SIZE    = 5_000_000

def download():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("gdown not installed — run: pip install gdown")
        sys.exit(1)

    # derronqi/yolov8-face — pose model with 5 facial landmarks
    # Gate C confirmed: keypoints shape [N, 5, 2], 117 FPS on T4
    url = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
    print(f"Downloading yolov8n-face.pt (derronqi, 5-kpt pose model)...")
    gdown.download(url, str(DEST), quiet=False)

    size = DEST.stat().st_size if DEST.exists() else 0
    if size < MIN_SIZE:
        DEST.unlink(missing_ok=True)
        print(f"FAILED: file too small ({size} bytes)")
        sys.exit(1)
    print(f"OK — {size/1e6:.1f} MB saved to {DEST}")

if __name__ == "__main__":
    download()
