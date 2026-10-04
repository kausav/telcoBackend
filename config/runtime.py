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
# A spec that passes the checks is served as soon as it exists; measured errors left in it are patched before that only while the
# design has taken less than this long (the refinement patches them afterwards anyway).
SPEC_DRAFT_REPAIR_SECONDS = max(0, int(os.getenv("SPEC_DRAFT_REPAIR_SECONDS", "90")))
# Background refinement of a spec that already works: the reviewer's findings go back to the designer for up to this long.
SPEC_REFINE_BUDGET_SECONDS = max(30, int(os.getenv("SPEC_REFINE_BUDGET_SECONDS", "900")))
# background (default): generation uses the first verified spec at once and a refinement replaces it when it is better;
# inline: a design is not offered until refinement has finished; off: no refinement.
SPEC_REFINE_MODE = os.getenv("SPEC_REFINE_MODE", "background").strip().lower()
if SPEC_REFINE_MODE not in {"background", "inline", "off"}:
    SPEC_REFINE_MODE = "background"
# The longest /scenario/generate waits for a spec that is still being designed before answering "not ready, retry".
SPEC_GENERATE_WAIT_SECONDS = max(1, int(os.getenv("SPEC_GENERATE_WAIT_SECONDS", "90")))
# A spec that has not been reviewed yet is the raw first design; /scenario/generate waits up to this long for the refinement's first
# improvement (or its end) before generating, so that the data comes from the reviewed spec. 0 generates from the first design at once.
SPEC_GENERATE_REFINE_WAIT_SECONDS = max(0, int(os.getenv("SPEC_GENERATE_REFINE_WAIT_SECONDS", "150")))
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
