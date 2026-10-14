"""Fixed application settings.

The service reads exactly four values from the environment, because they are secrets or belong to one deployment:

    GEMINI_API_KEY       the Gemini API key
    MONGODB_URI          the MongoDB connection string
    MONGODB_DB_NAME      the database name (default below)
    API_STATIC_TOKENS    the static API token(s), comma-separated so a token can be rotated; "off" disables authentication

Everything else lives here. Change a value in this file and redeploy; nothing else needs setting. No secret belongs in this file.
"""
from __future__ import annotations

# ---- Gemini ----------------------------------------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_TIMEOUT_MS = 60_000
GEMINI_RETRY_ATTEMPTS = 1                      # 1..6

# ---- MongoDB (the URI and database name come from the environment) ------------------------------------------------
MONGODB_DEFAULT_URI = "mongodb://localhost:27017"      # used only when MONGODB_URI is not set (local development)
MONGODB_DEFAULT_DB_NAME = "telco_scenario_db"          # used only when MONGODB_DB_NAME is not set
MONGODB_APP_NAME = "telco-agentic-sdg"
MONGODB_SERVER_SELECTION_TIMEOUT_MS = 5_000
MONGODB_CONNECT_TIMEOUT_MS = 5_000
MONGODB_SOCKET_TIMEOUT_MS = 10_000
COLLECTIONS = {
    "scenarios": "scenarios",
    "drafts": "scenario_drafts",
    "scenario_variables": "scenario_variables",
    "user_variables": "scenario_user_variables",
    "selection_locks": "scenario_selection_locks",
    "industry_sources": "industry_source_documents",
    "generation_specs": "generation_specs",
    "generation_spec_claims": "generation_spec_claims",
    "domain_lexicon": "domain_lexicon",
    "country_limits": "country_operating_limits",     # rail floors/caps, fees and cash denominations per country
}
DRAFT_TTL_SECONDS = 86_400

# ---- HTTP / API ------------------------------------------------------------------------------------------------------
CORS_ALLOW_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]   # add the UI's origin here when it is served elsewhere
API_AUTH_PUBLIC_PATHS = ("/", "/health", "/ready")                       # never need the token (probes)
API_AUTH_PROTECT_DOCS = False                                            # True: the interactive docs need the token as well
RUNTIME_DATA_DIR = "runtime_data"                                        # relative to the project root, or absolute
MAX_CSV_BYTES = 10 * 1024 * 1024
INDUSTRY_SOURCE_MAX_JSON_BYTES = 8 * 1024 * 1024
SEED_REFERENCE_DATA = True                                               # insert the bundled catalog / lexicon rows MongoDB lacks
DOMAIN_LEXICON_CACHE_SECONDS = 60
COUNTRY_LIMITS_CACHE_SECONDS = 60                                        # how long a country's rail limits are kept before MongoDB is read again

# ---- Caches ----------------------------------------------------------------------------------------------------------
RUNTIME_CACHE_MAX_ITEMS = 256
PROPOSAL_CACHE_TTL_SECONDS = 1_800
SOURCE_CATALOG_CACHE_TTL_SECONDS = 60

# ---- Public egress IP diagnostics (shown to help with MongoDB IP allow-lists) -----------------------------------------
PUBLIC_EGRESS_IP = ""                                                    # set to skip the lookup
PUBLIC_EGRESS_IP_URL = "https://api.ipify.org?format=text"
PUBLIC_EGRESS_IP_TIMEOUT_SECONDS = 2.5
PUBLIC_EGRESS_IP_CACHE_TTL_SECONDS = 900.0

# ---- Proposal (variable selection) -----------------------------------------------------------------------------------
JSON_SOURCE_LLM_CATALOG_LIMIT = 1_200
JSON_SOURCE_LLM_FIELDS_PER_MODEL = 12
SCHEMA_MAX_VARIABLES = 60                      # variables per proposal, DB variables included
SCHEMA_MIN_VARIABLE_SCORE = 42.0
PROPOSE_LLM_ADVISOR = True

# ---- Generation spec design ------------------------------------------------------------------------------------------
SPEC_LLM_TIMEOUT_MS = 240_000                  # one model call that designs a spec
SPEC_REVIEW = True                             # a second model reads a sample of the data and reports what looks wrong
SPEC_COMPILE_BUDGET_SECONDS = 1_200            # the longest a design may take across its repair rounds
SPEC_REFINE_MODE = "background"                # background | inline | off
SPEC_GENERATE_WAIT_SECONDS = 20                # how long one /scenario/generate request waits for an unfinished spec
SPEC_DESIGN_SECONDS = 720                      # no further model round is begun that would end beyond this
SPEC_SETTLE_SECONDS = 180                      # a spec with open errors is refined at most this long before it is served
SPEC_SERVE_EARLY = True                        # serve a first spec with no open error at once
SPEC_FIRST_SERVE_SECONDS = 420                 # repair rounds before the first serve may run only inside this window
SPEC_WARM_ON_PROPOSE = True                    # start designing the spec as soon as a proposal is made
SPEC_DRAFT_HARD_SECONDS = 480                  # no model call of the first design (author, expectations, repair, review) runs past this; a call is cut to fit

# ---- Generation and quality ------------------------------------------------------------------------------------------
MIN_LOGIC_QUALITY_PERCENT = 90.0
OPEN_BOUND_SPAN_INT = 30                       # span used when a numeric range has a lower bound but no upper bound
OPEN_BOUND_SPAN_FLOAT = 500.0
GENERATION_MAX_ATTEMPTS_PER_RECORD = 8         # 1..20
AGENTIC_REQUIRE_CLEAN_RECORDS = True
AGENTIC_REQUIRE_EXACT_RECORD_COUNT = True
AGENTIC_LLM_QA_MODE = "off"                    # semantic Gemini QA is opt-in: it makes one request per record chunk
QA_LLM_MODE = "off"
QA_LLM_CHUNK_SIZE = 10
