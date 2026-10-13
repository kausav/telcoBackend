# Python 3.13 on Chainguard's Wolfi base.
#
# Why not chainguard/python:latest?  It moved to Python 3.15, and `ormsgpack`
# (a hard dependency of langgraph) does not build on 3.15 yet, so the install
# fails. Wolfi lets us choose the Python version ourselves.
#
# Binutils / ld / compilers exist only in the build stage and are not shipped.

# ---- build stage ----
FROM cgr.dev/chainguard/wolfi-base:latest AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apk add --no-cache python-3.13 py3.13-pip build-base

WORKDIR /app
COPY requirements.txt .
RUN python3.13 -m pip install --target=/app/deps -r requirements.txt

# ---- runtime stage: python only, no compilers, no binutils ----
FROM cgr.dev/chainguard/wolfi-base:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/deps

RUN apk add --no-cache python-3.13

WORKDIR /app
COPY --from=builder /app/deps /app/deps
COPY --chown=65532:65532 . .

# The app creates /app/runtime_data at start-up, so the non-root user must be
# able to write there (/app itself is owned by root, which caused the failure).
RUN mkdir -p /app/runtime_data && chown 65532:65532 /app /app/runtime_data

USER 65532:65532
# Cloud Run injects PORT (8080); locally it falls back to 8000.
EXPOSE 8080

CMD ["/bin/sh", "-c", "exec python3.13 -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
