import base64
import os
from io import BytesIO
from typing import Any, Dict

import runpod
from PIL import Image
from vllm import LLM, SamplingParams

model = None


def _configure_cache_dirs() -> None:
    """Prefer caching to a mounted network volume when available."""
    volume_root = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
    if not os.path.isdir(volume_root):
        return

    cache_root = os.path.join(volume_root, "cache")
    hf_home = os.getenv("HF_HOME") or os.path.join(cache_root, "hf")
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE") or os.path.join(hf_home, "hub")
    vllm_cache = os.getenv("VLLM_CACHE_ROOT") or os.path.join(cache_root, "vllm")

    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(vllm_cache, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub_cache)
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("VLLM_CACHE_ROOT", vllm_cache)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def initialize_model() -> LLM:
    global model
    if model is not None:
        return model
    _configure_cache_dirs()
    model = LLM(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
        trust_remote_code=True,
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
        # vLLM offline multimodal needs a per-request image limit configured up-front.
        limit_mm_per_prompt={"image": 1},
    )
    return model


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if model is None:
            try:
                initialize_model()
            except Exception as exc:  # noqa: BLE001
                return {"error": f"model_init_failed: {exc}", "status": "error"}

        data = job.get("input", {})
        user_prompt = data.get("prompt", "Describe this image")
        image_b64 = data.get("image", "")
        if not image_b64:
            return {"error": "image is required", "status": "error"}

        # Offline vLLM multimodal path: provide a prompt with <image> placeholders and
        # attach the decoded PIL.Image via multi_modal_data.
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        # Force full decode early; avoid downstream "broken data stream" errors.
        image.load()
        image = image.convert("RGB")
        prompt = f"USER: <image>\n{user_prompt}\nASSISTANT:"

        sampling = SamplingParams(
            max_tokens=data.get("max_tokens", 256),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
        )
        outputs = model.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            sampling,
        )
        return {"output": outputs[0].outputs[0].text, "status": "success"}
    except Exception as exc:
        return {"error": str(exc), "status": "error"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
