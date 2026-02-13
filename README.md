# Second Endpoint (Vision, vLLM)

RunPod serverless vision endpoint for `Qwen/Qwen3-VL-8B-Instruct`.

## Build

```bash
./scripts/build.sh
```

## Push

```bash
./scripts/push.sh
```

## RunPod settings

- GPU: NVIDIA L4
- Workers: min `0`, max `2`
- Idle timeout: `60s`
- Env:
  - `MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct`
  - `MAX_MODEL_LEN=4096`
  - `TRUST_REMOTE_CODE=True`
  - `GPU_MEMORY_UTILIZATION=0.9`
  - `HF_TOKEN=<token>`
