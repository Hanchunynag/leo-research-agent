# Keep the dependency installer pinned and sourced from the official uv image.
# Installing the unpinned `uv` package through pip made the image build depend
# on a transient registry artifact and failed hash verification on arm64.
FROM ghcr.io/astral-sh/uv:0.11.30 AS uv
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    LEO_MINERU_EXECUTABLE=/opt/mineru/bin/mineru \
    LEO_PADDLEOCR_EXECUTABLE=/opt/paddleocr/bin/python

WORKDIR /app
COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# MinerU must live in a Linux virtual environment inside the image. Runtime
# data is stored in Docker volumes, so using a workspace-local venv would
# accidentally execute a host (often macOS) interpreter if one were present.
RUN --mount=type=cache,id=leo-uv-cache,target=/root/.cache/uv \
    uv venv --python 3.11 /opt/mineru \
    && UV_HTTP_RETRIES=10 UV_HTTP_TIMEOUT=600 \
        uv pip install --python /opt/mineru/bin/python \
        "mineru[core]==3.4.4" \
        "torch==2.13.0" \
        "torchvision==0.28.0" \
    && uv pip uninstall --python /opt/mineru/bin/python --yes opencv-python \
    && uv pip install --python /opt/mineru/bin/python --no-deps "opencv-python-headless==5.0.0.93"

# PaddleOCR is intentionally isolated from the application's uv environment
# and from runtime data volumes. The workers are launched with this
# interpreter explicitly, so a host macOS venv can never be selected inside
# the Linux container. Keep both the table pipeline and PaddleOCR-VL in the
# image because the parsing pipeline uses them as its local fallback engines.
RUN --mount=type=cache,id=leo-uv-cache,target=/root/.cache/uv \
    uv venv --python 3.11 /opt/paddleocr \
    && UV_HTTP_RETRIES=10 UV_HTTP_TIMEOUT=600 \
        uv pip install --python /opt/paddleocr/bin/python \
        "paddlepaddle==3.2.1" \
        "paddleocr==3.7.0" \
        "paddlex==3.7.2" \
        "beautifulsoup4" \
        "einops" \
        "ftfy" \
        "imagesize" \
        "Jinja2" \
        "latex2mathml" \
        "lxml" \
        "openpyxl" \
        "premailer" \
        "python-bidi" \
        "regex" \
        "safetensors>=0.7.0" \
        "scikit-learn" \
        "scipy" \
        "sentencepiece" \
        "shapely" \
        "tiktoken" \
        "tokenizers>=0.19" \
    && (uv pip uninstall --python /opt/paddleocr/bin/python --yes opencv-contrib-python || true) \
    && uv pip install --python /opt/paddleocr/bin/python --no-deps \
        "opencv-contrib-python-headless==4.10.0.84" \
    && /opt/paddleocr/bin/python - <<'PY'
from paddleocr import PaddleOCR, PaddleOCRVL, TableRecognitionPipelineV2
import paddle
import paddleocr
from importlib import metadata
from importlib.util import find_spec

assert paddle.__version__ == "3.2.1", paddle.__version__
assert paddleocr.__version__ == "3.7.0", paddleocr.__version__
assert find_spec("cv2") is not None, "OpenCV is not available"
required = [
    "beautifulsoup4", "einops", "ftfy", "imagesize", "Jinja2",
    "latex2mathml", "lxml", "openpyxl", "premailer", "python-bidi",
    "regex", "safetensors", "scikit-learn", "scipy", "sentencepiece",
    "shapely", "tiktoken", "tokenizers",
]
missing = []
for name in required:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        missing.append(name)
assert not missing, f"PaddleX OCR dependencies missing: {missing}"
print("PaddleOCR Docker import check: ok")
print(f"paddle={paddle.__version__} paddleocr={paddleocr.__version__}")
print(PaddleOCR.__name__, TableRecognitionPipelineV2.__name__, PaddleOCRVL.__name__)
PY

COPY app ./app
COPY scripts ./scripts
COPY skills ./skills
COPY main.py README.md .env.example ./

EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.web.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
