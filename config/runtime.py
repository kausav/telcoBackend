"""Application settings.

The environment supplies only the four values in ``config.constants`` (GEMINI_API_KEY, MONGODB_URI, MONGODB_DB_NAME,
API_STATIC_TOKENS); every other setting is a constant in ``config.constants`` and is re-exported here under the same names.
Industry/domain standards are stored exclusively in MongoDB; this module has no filesystem standards directory.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from config import constants as _c

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def resolve_path(value: str | None, default: Path) -> Path:
    candidate = Path(value).expanduser() if value else default
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


RUNTIME_DATA_DIR = resolve_path(_c.RUNTIME_DATA_DIR, ROOT / "runtime_data")
try:
    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    raise RuntimeError(
        f"Runtime data directory is not writable or could not be created: {RUNTIME_DATA_DIR}. "
        "Change RUNTIME_DATA_DIR in config/constants.py to a directory the service account can write to."
    ) from exc

CORS_ALLOW_ORIGINS = list(_c.CORS_ALLOW_ORIGINS)

# Static API token(s) from the environment (API_STATIC_TOKENS), comma-separated so a token can be rotated without downtime. Callers send
# "Authorization: Bearer <token>" or "X-API-Key: <token>". Unset: authentication stays ON and no token is accepted (every protected
# request is refused until the variable is set). "off" switches authentication off (local development only).
_raw_tokens = os.getenv("API_STATIC_TOKENS", "").strip()
API_AUTH_DISABLED = _raw_tokens.lower() in {"off", "none", "disabled"}
API_STATIC_TOKENS = [] if API_AUTH_DISABLED else [t.strip() for t in _raw_tokens.split(",") if t.strip()]
API_AUTH_PUBLIC_PATHS = tuple(_c.API_AUTH_PUBLIC_PATHS)
API_AUTH_PROTECT_DOCS = _c.API_AUTH_PROTECT_DOCS

MAX_CSV_BYTES = _c.MAX_CSV_BYTES
INDUSTRY_SOURCE_MAX_JSON_BYTES = _c.INDUSTRY_SOURCE_MAX_JSON_BYTES
JSON_SOURCE_LLM_CATALOG_LIMIT = _c.JSON_SOURCE_LLM_CATALOG_LIMIT
# Number of representative fields shown per source business model to Gemini. The complete
# MongoDB catalog is still used by deterministic compilation; this only keeps propose latency low.
JSON_SOURCE_LLM_FIELDS_PER_MODEL = max(3, _c.JSON_SOURCE_LLM_FIELDS_PER_MODEL)
# Process-local cache for immutable active source catalogs. Admin mutations invalidate the cache.
SOURCE_CATALOG_CACHE_TTL_SECONDS = max(5, _c.SOURCE_CATALOG_CACHE_TTL_SECONDS)

# Total variables per proposal, DB variables included (they are always kept first; JSON variables fill the rest).
SCHEMA_MAX_VARIABLES = max(1, _c.SCHEMA_MAX_VARIABLES)
# When true, the LLM names the relevant source resources/variables once per distinct input and that choice is
# locked in MongoDB, so later runs of the same scenario reuse it. When false no model is called at all and the
# JSON variables come only from the resources the DB variables belong to.
PROPOSE_LLM_ADVISOR = _c.PROPOSE_LLM_ADVISOR
SCHEMA_MIN_VARIABLE_SCORE = _c.SCHEMA_MIN_VARIABLE_SCORE

# A decision threshold for the logic quality of generated data (percent of records that satisfy every scenario rule).
MIN_LOGIC_QUALITY_PERCENT = min(100.0, max(0.0, _c.MIN_LOGIC_QUALITY_PERCENT))

# Time allowed for one model call that designs a scenario's generation spec (the spec is long, so this is generous).
SPEC_LLM_TIMEOUT_MS = max(10000, _c.SPEC_LLM_TIMEOUT_MS)
# A second model reviews a sample of the data a designed spec produces and sends what a person would call wrong back to the
# designer. Set SPEC_REVIEW to False in config/constants.py to design with the deterministic checks only.
SPEC_REVIEW = _c.SPEC_REVIEW
# Total time one design may take across its repair rounds; once spent, the spec so far is rejected rather than waiting on.
SPEC_COMPILE_BUDGET_SECONDS = max(30, _c.SPEC_COMPILE_BUDGET_SECONDS)
# background (default): the design (first spec, then its refinement) runs in the background, and a request that needs the spec waits
# for it to settle, so every request for a scenario generates from the same final spec; inline: refinement runs in the caller's
# thread; off: no refinement (the first verified spec is final).
SPEC_REFINE_MODE = str(_c.SPEC_REFINE_MODE).strip().lower()
if SPEC_REFINE_MODE not in {"background", "inline", "off"}:
    SPEC_REFINE_MODE = "background"
# How long ONE /scenario/generate request is held open for a spec that is still being designed or refined. It is short on purpose:
# a design takes minutes (2 to 6 in practice), and a request held that long is cut off by clients and proxies long before. A request
# that finds the spec unfinished answers 503 + Retry-After at once (the design carries on in the background) and the caller asks again.
SPEC_GENERATE_WAIT_SECONDS = max(1, _c.SPEC_GENERATE_WAIT_SECONDS)
# How long one design may take, draft and refinement together, counted from when it began. No further model round is begun that is
# expected to end beyond it: a repair of the first spec, the patch made while the first spec is checked, and every refinement round.
# It is independent of the request wait above: nobody holds a request open for it, so it can be long enough for the reviewer's
# findings to be repaired (a refinement round needs one to two minutes) instead of shipping with the errors still open.
SPEC_DESIGN_SECONDS = max(30, _c.SPEC_DESIGN_SECONDS)
# A spec that is still being refined is not served (generate answers "refining, retry") until its refinement has run this long;
# after that the best revision so far is served. The safety net for a refinement that never ends.
SPEC_SETTLE_SECONDS = max(0, _c.SPEC_SETTLE_SECONDS)
# A spec whose first version has no open error (the deterministic checks found nothing impossible) is served at once, while its
# refinement carries on; later requests for the same scenario then get the improved revision. Turn off to make every request wait
# (up to SPEC_SETTLE_SECONDS) for the refinement to finish.
SPEC_SERVE_EARLY = _c.SPEC_SERVE_EARLY
# The first spec is served as soon as it exists. The patch rounds that repair its measured errors before it is served may run only
# while the design is younger than this, because each takes about as long as the first design: past it, the repair moves to the
# background refinement and the caller is not kept waiting for it.
SPEC_FIRST_SERVE_SECONDS = max(30, _c.SPEC_FIRST_SERVE_SECONDS)
SPEC_DRAFT_HARD_SECONDS = max(60, _c.SPEC_DRAFT_HARD_SECONDS)
# Start designing a proposal's spec in the background as soon as it is proposed (the person reviews it meanwhile).
SPEC_WARM_ON_PROPOSE = _c.SPEC_WARM_ON_PROPOSE

# Span used when a curated numeric range declares a lower bound but no upper bound (counts / amounts).
OPEN_BOUND_SPAN_INT = max(1, _c.OPEN_BOUND_SPAN_INT)
OPEN_BOUND_SPAN_FLOAT = max(1.0, _c.OPEN_BOUND_SPAN_FLOAT)
GENERATION_MAX_ATTEMPTS_PER_RECORD = max(1, min(20, _c.GENERATION_MAX_ATTEMPTS_PER_RECORD))
AGENTIC_REQUIRE_CLEAN_RECORDS = _c.AGENTIC_REQUIRE_CLEAN_RECORDS
AGENTIC_REQUIRE_EXACT_RECORD_COUNT = _c.AGENTIC_REQUIRE_EXACT_RECORD_COUNT
# Semantic Gemini QA is synchronous and makes one request per record chunk. Keep it opt-in so
# large generate requests do not wait on dozens of sequential provider calls.
AGENTIC_LLM_QA_MODE = str(_c.AGENTIC_LLM_QA_MODE).strip().lower()
