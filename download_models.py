"""Download the AI model files into ./models (run once before building the exe).

  models/face_detection_yunet_2023mar.onnx  - finds faces + mouth corners (OpenCV Zoo, MIT licence)
  models/genderage.onnx                     - guesses male/female look of a face (InsightFace buffalo_l,
                                              non-commercial research licence - see context.md)
  models/pyannote_segmentation_3.onnx       - finds overlapping speech (pyannote, MIT)
  models/whisper-small/                     - Whisper "small" speech model for language detection (MIT)
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "models"
ROOT.mkdir(exist_ok=True)

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def fetch(url: str) -> bytes:
    print("Downloading", url)
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def main() -> None:
    yunet = ROOT / "face_detection_yunet_2023mar.onnx"
    if not yunet.exists():
        yunet.write_bytes(fetch(YUNET_URL))

    gender = ROOT / "genderage.onnx"
    if not gender.exists():
        data = fetch(BUFFALO_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.endswith("genderage.onnx"))
            gender.write_bytes(z.read(name))

    seg = ROOT / "pyannote_segmentation_3.onnx"
    if not seg.exists():
        from huggingface_hub import hf_hub_download
        import shutil

        shutil.copy(hf_hub_download("onnx-community/pyannote-segmentation-3.0", "onnx/model.onnx"), seg)

    wdir = ROOT / "whisper-small"
    if not (wdir / "model.bin").exists():
        from huggingface_hub import snapshot_download

        snapshot_download("Systran/faster-whisper-small", local_dir=str(wdir),
                          allow_patterns=["config.json", "model.bin", "tokenizer.json",
                                          "vocabulary.*", "preprocessor_config.json"])
    print("Models ready in", ROOT)
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
