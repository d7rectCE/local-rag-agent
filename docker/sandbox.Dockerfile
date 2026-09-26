# Sandbox image for the code agent (ТЗ ч.2 S17, NFR8).
# The container runs without network, as uid 1000, with a read-only root file system;
# only the task's working copy is writable. Package versions follow the project environment,
# so the demo corpus code and notebooks run the same way inside.
#
#   docker build -t rag-sandbox:py312 -f docker/sandbox.Dockerfile docker
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        numpy==2.5.3 scipy==1.18.1 pandas==3.0.5 scikit-learn==1.9.1 matplotlib==3.11.2 \
        nbformat==5.11.1 nbclient==0.11.0 ipykernel==7.3.0 pytest==9.1.1 pyflakes \
    && useradd --create-home --uid 1000 sandbox \
    && python -m ipykernel install --name python3 --prefix /usr/local

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter \
    IPYTHONDIR=/tmp/ipython \
    MPLCONFIGDIR=/tmp/matplotlib

USER 1000
WORKDIR /work
