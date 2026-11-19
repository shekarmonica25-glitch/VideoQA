"""Builds dist/VideoQA/VideoQA.exe (a folder you can copy to any Windows PC).

    .venv\\Scripts\\python.exe download_models.py   (once)
    .venv\\Scripts\\python.exe build_exe.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "VideoQA"


def main() -> None:
    models = ROOT / "models"
    need = ["face_detection_yunet_2023mar.onnx", "genderage.onnx", "pyannote_segmentation_3.onnx",
            "whisper-small/model.bin"]
    missing = [n for n in need if not (models / n).exists()]
    if missing:
        sys.exit(f"Missing models {missing} - run download_models.py first.")

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--name", "VideoQA",
           "--collect-data", "faster_whisper", "--collect-binaries", "ctranslate2",
           "--collect-binaries", "onnxruntime", "--collect-submodules", "av",
           "--exclude-module", "onnx", "--exclude-module", "matplotlib", "--exclude-module", "imageio_ffmpeg",
           "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"), "--specpath", str(ROOT / "build"),
           str(ROOT / "app.py")]
    subprocess.check_call(cmd)

    # models next to the exe (not inside _internal) so they can be swapped without rebuilding
    dst = DIST / "models"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for n in need[:3]:
        shutil.copy2(models / n, dst / n)
    (dst / "whisper-small").mkdir()
    for f in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt", "preprocessor_config.json"):
        if (models / "whisper-small" / f).exists():
            shutil.copy2(models / "whisper-small" / f, dst / "whisper-small" / f)

    # ffmpeg.exe in its own folder next to the exe
    import imageio_ffmpeg

    (DIST / "ffmpeg").mkdir(exist_ok=True)
    shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), DIST / "ffmpeg" / "ffmpeg.exe")
    shutil.copy2(ROOT / "README_USER.txt", DIST / "README.txt")

    size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file()) / 1e6
    print(f"\nDone: {DIST / 'VideoQA.exe'}  (folder size {size:.0f} MB)")


if __name__ == "__main__":
    main()
