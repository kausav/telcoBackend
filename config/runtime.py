"""Environment-backed application settings.

Industry/domain standards are stored exclusively in MongoDB. This module contains
only generic application/runtime settings; it intentionally has no filesystem
standards directory or bundled industry-standard JSON configuration.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def resolve_path(value: str | None, default: Path) -> Path:
    candidate = Path(value).expanduser() if value else default
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


RUNTIME_DATA_DIR = resolve_path(os.getenv("RUNTIME_DATA_DIR"), ROOT / "runtime_data")
try:
    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    raise RuntimeError(
        f"Runtime data directory is not writable or could not be created: {RUNTIME_DATA_DIR}. "
        "Set RUNTIME_DATA_DIR to a writable directory for the service account."
    ) from exc

CORS_ALLOW_ORIGINS = [
    item.strip()
    for item in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    if item.strip()
]

MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(10 * 1024 * 1024)))
INDUSTRY_SOURCE_MAX_JSON_BYTES = int(os.getenv("INDUSTRY_SOURCE_MAX_JSON_BYTES", str(8 * 1024 * 1024)))
JSON_SOURCE_LLM_CATALOG_LIMIT = int(os.getenv("JSON_SOURCE_LLM_CATALOG_LIMIT", "1200"))
# Number of representative fields shown per source business model to Gemini. The complete
# MongoDB catalog is still used by deterministic compilation; this only keeps propose latency low.
JSON_SOURCE_LLM_FIELDS_PER_MODEL = max(3, int(os.getenv("JSON_SOURCE_LLM_FIELDS_PER_MODEL", "12")))
# Process-local cache for immutable active source catalogs. Admin mutations invalidate the cache.
SOURCE_CATALOG_CACHE_TTL_SECONDS = max(5, int(os.getenv("SOURCE_CATALOG_CACHE_TTL_SECONDS", "60")))

# Total variables per proposal, DB variables included (they are always kept first; JSON variables fill the rest).
SCHEMA_MAX_VARIABLES = max(1, int(os.getenv("SCHEMA_MAX_VARIABLES", "60")))
# When true, the LLM names the relevant source resources/variables once per distinct input and that choice is
# locked in MongoDB, so later runs of the same scenario reuse it. When false no model is called at all and the
# JSON variables come only from the resources the DB variables belong to.
PROPOSE_LLM_ADVISOR = os.getenv("PROPOSE_LLM_ADVISOR", "true").strip().lower() in {"1", "true", "yes", "on"}
SCHEMA_MIN_VARIABLE_SCORE = float(os.getenv("SCHEMA_MIN_VARIABLE_SCORE", "42"))

# A decision threshold for the logic quality of generated data (percent of records that satisfy every scenario rule).
MIN_LOGIC_QUALITY_PERCENT = min(100.0, max(0.0, float(os.getenv("MIN_LOGIC_QUALITY_PERCENT", "90"))))
# Time allowed for one model call that designs a scenario's generation spec (the spec is long, so this is generous).
SPEC_LLM_TIMEOUT_MS = max(10000, int(os.getenv("SPEC_LLM_TIMEOUT_MS", "240000")))
# A second model reviews a sample of the data a designed spec produces and sends what a person would call wrong back to the
# designer. Turn off to design with the deterministic checks only.
SPEC_REVIEW = os.getenv("SPEC_REVIEW", "true").strip().lower() in {"1", "true", "yes", "on"}
# Total time one design may take across its repair rounds; once spent, the spec so far is rejected rather than waiting on.
SPEC_COMPILE_BUDGET_SECONDS = max(30, int(os.getenv("SPEC_COMPILE_BUDGET_SECONDS", "600")))
# background (default): the design (first spec, then its refinement) runs in the background, and a request that needs the spec waits
# for it to settle, so every request for a scenario generates from the same final spec; inline: refinement runs in the caller's
# thread; off: no refinement (the first verified spec is final).
SPEC_REFINE_MODE = os.getenv("SPEC_REFINE_MODE", "background").strip().lower()
if SPEC_REFINE_MODE not in {"background", "inline", "off"}:
    SPEC_REFINE_MODE = "background"
# The longest /scenario/generate waits for a spec that is still being designed before answering "not ready, retry".
SPEC_GENERATE_WAIT_SECONDS = max(1, int(os.getenv("SPEC_GENERATE_WAIT_SECONDS", "240")))
# A design is the first verified spec followed by its refinement, and a request waits for both, so that the data it gets is the data
# every later request gets. The refinement starts no further round once this long has passed since the design began (a round is
# expected to take as long as the last author call plus a review); the default keeps 30 s of the wait above in hand.
SPEC_REFINE_BUDGET_SECONDS = max(30, int(os.getenv("SPEC_REFINE_BUDGET_SECONDS", str(max(30, SPEC_GENERATE_WAIT_SECONDS - 30)))))
# Measured errors left in the first spec are patched right away (the reviewer reads it at the same time) only when that patch is
# expected to be done within this long since the design began; otherwise the refinement patches them. Default: half of the wait.
SPEC_DRAFT_REPAIR_SECONDS = max(0, int(os.getenv("SPEC_DRAFT_REPAIR_SECONDS", str(int(SPEC_GENERATE_WAIT_SECONDS * 0.5)))))
# Start designing a proposal's spec in the background as soon as it is proposed (the person reviews it meanwhile).
SPEC_WARM_ON_PROPOSE = os.getenv("SPEC_WARM_ON_PROPOSE", "true").strip().lower() in {"1", "true", "yes", "on"}

# Span used when a curated numeric range declares a lower bound but no upper bound (counts / amounts).
OPEN_BOUND_SPAN_INT = max(1, int(os.getenv("OPEN_BOUND_SPAN_INT", "30")))
OPEN_BOUND_SPAN_FLOAT = max(1.0, float(os.getenv("OPEN_BOUND_SPAN_FLOAT", "500")))
GENERATION_MAX_ATTEMPTS_PER_RECORD = max(1, min(20, int(os.getenv("GENERATION_MAX_ATTEMPTS_PER_RECORD", "8"))))
AGENTIC_REQUIRE_CLEAN_RECORDS = os.getenv("AGENTIC_REQUIRE_CLEAN_RECORDS", "true").strip().lower() in {"1", "true", "yes", "on"}
AGENTIC_REQUIRE_EXACT_RECORD_COUNT = os.getenv("AGENTIC_REQUIRE_EXACT_RECORD_COUNT", "true").strip().lower() in {"1", "true", "yes", "on"}
# Semantic Gemini QA is synchronous and makes one request per record chunk. Keep it opt-in so
# large generate requests do not wait on dozens of sequential provider calls.
AGENTIC_LLM_QA_MODE = os.getenv("AGENTIC_LLM_QA_MODE", "off").strip().lower()
