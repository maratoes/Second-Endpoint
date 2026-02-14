#!/usr/bin/env python3
"""Pod smoke-test for the vision endpoint image."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure repo root (where handler.py lives) is importable even if CWD != /app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _maybe_prefetch(model_name: str) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        print("[pod_smoke] huggingface_hub not available; skipping prefetch", flush=True)
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(f"[pod_smoke] prefetch start model={model_name}", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=model_name,
        token=token,
        resume_download=True,
    )
    print(f"[pod_smoke] prefetch done in {time.time()-t0:.1f}s", flush=True)


def main() -> int:
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    _maybe_prefetch(model_name)

    import handler as h

    print("[pod_smoke] initializing model...", flush=True)
    t0 = time.time()
    h.initialize_model()
    print(f"[pod_smoke] model initialized in {time.time()-t0:.1f}s", flush=True)

    # 1x1 transparent PNG base64
    tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9oN7LxkAAAAASUVORK5CYII="
    payload = {
        "input": {
            "prompt": "Describe this image in one short sentence.",
            "image": tiny_png,
            "max_tokens": 32,
            "temperature": 0.1,
        }
    }
    print("[pod_smoke] running inference...", flush=True)
    out = h.handler(payload)
    print("[pod_smoke] output:", out, flush=True)
    return 0 if isinstance(out, dict) and out.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
