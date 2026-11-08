"""Run faster-whisper outside the Qt process.

The parent process receives JSON on stdout. Any native CTranslate2 failure is
therefore reported as a subprocess exit instead of closing the GUI silently.
"""

import json
import os
import sys
from pathlib import Path


def main():
    request = json.loads(sys.argv[1])
    if request.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = request["hf_endpoint"]

    from faster_whisper import WhisperModel

    root = Path(__file__).resolve().parent
    device = request.get("device") or "cpu"
    compute_type = request.get("compute_type") or "int8"
    if device == "auto":
        # Prefer the conservative CPU path for the first stable MVP.
        device = "cpu"
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"

    model_source = request.get("model_path") or request.get("model", "small")
    recognizer = WhisperModel(
        model_source,
        device=device,
        compute_type=compute_type,
        download_root=str(root / request.get("model_download_root", "models")),
    )
    segments, _info = recognizer.transcribe(
        request["wav_path"],
        language=request.get("language", "zh"),
        vad_filter=True,
        beam_size=5,
    )
    text = "".join(segment.text for segment in segments).strip()
    print(json.dumps({"text": text}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
        raise
