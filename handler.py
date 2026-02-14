import base64
import os
from io import BytesIO
from typing import Any, Dict

import runpod
from PIL import Image
from vllm import LLM, SamplingParams

model = None


def initialize_model() -> LLM:
    global model
    if model is not None:
        return model
    model = LLM(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
        trust_remote_code=True,
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
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
        prompt = data.get("prompt", "Describe this image")
        image_b64 = data.get("image", "")
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        sampling = SamplingParams(
            max_tokens=data.get("max_tokens", 256),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
        )
        outputs = model.chat(messages, sampling)
        return {"output": outputs[0].outputs[0].text, "status": "success"}
    except Exception as exc:
        return {"error": str(exc), "status": "error"}


runpod.serverless.start({"handler": handler})
