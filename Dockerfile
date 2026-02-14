FROM runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204

RUN pip install --no-cache-dir vllm==0.15.1 runpod==1.7.0 pillow requests

WORKDIR /app
COPY handler.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
ENV MAX_MODEL_LEN=4096
ENV TRUST_REMOTE_CODE=True
ENV GPU_MEMORY_UTILIZATION=0.9

CMD ["python", "-u", "handler.py"]
