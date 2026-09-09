# Federation console + training, in one image.
#
# Both live here on purpose: the dashboard streams a run by tailing an
# append-only log on its own filesystem, so a dashboard that cannot see the
# training process cannot show a live federation. Splitting them into two
# containers would leave the live console permanently empty.
#
# Linux is the better host for this anyway -- Ray is markedly more reliable here
# than on Windows, where most of this project's process-management pain came from.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The flwr CLI prints an emoji and dies on a non-UTF-8 console.
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    DATA_ROOT=/data \
    RESULTS_DIR=/app/fl_rnaseq/results

# procps supplies pkill, which the launcher's stop() uses to tear down a run's
# process tree. curl is for the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first, from the CPU index: the default index would pull the CUDA build,
# roughly 2 GB of wheels that cannot be used on a CPU-only VM.
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

COPY fl_rnaseq/pyproject.toml /app/fl_rnaseq/pyproject.toml
COPY fl_rnaseq/README.md* /app/fl_rnaseq/
COPY fl_rnaseq/LICENSE /app/fl_rnaseq/LICENSE
COPY fl_rnaseq/fl_rnaseq /app/fl_rnaseq/fl_rnaseq
RUN pip install --no-cache-dir -e /app/fl_rnaseq

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir -r /app/webapp/requirements.txt

COPY fl_rnaseq/scripts /app/fl_rnaseq/scripts
COPY webapp /app/webapp
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The dataset is NOT baked in: it is ~200 MB of licensed UCI data that would
# bloat every layer. entrypoint.sh fetches it on first boot into a volume.
VOLUME ["/data", "/app/fl_rnaseq/results"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
