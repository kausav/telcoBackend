"""Production entrypoint for durable scenario generation workers."""
from __future__ import annotations

import logging
import os
import signal
import threading

from models.database import ping
from models.generation_job import GenerationJobModel
from core.generation_jobs import run_worker


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    ping()
    GenerationJobModel.ensure_indexes()
    worker_id = os.getenv("GENERATION_WORKER_ID")
    stop_event = threading.Event()

    def _stop(_signum, _frame):
        logger.info("Generation worker %s received shutdown signal", worker_id or "auto")
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    run_worker(worker_id=worker_id, stop_event=stop_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
