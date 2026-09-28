# Industry Source Registry — MongoDB Only

## Source-of-truth rule

Industry/domain standard JSON documents are **only** stored and read from MongoDB. Runtime code does not load them from a `resources/` directory, `config/telecom_standards/`, a downloaded standards cache, a telecom registry, external URLs, or any other filesystem source.

For a request identified by `industryType + domain`, generation/proposal uses:

1. Active JSON source documents in the MongoDB `industry_source_documents` collection; and/or
2. Explicitly persisted MongoDB `scenario_variables` definitions for the requested scenario.

If neither source exists, `/scenario/propose` and `/scenario/generate` fail closed.

## Upload API

`POST /industry-sources/upload` accepts exactly three multipart inputs:

- `industryType` — one value for the entire batch
- `domain` — one value for the entire batch
- `file` — one or more JSON files

Example:

```bash
curl --location 'http://localhost:8000/industry-sources/upload' \
  --form 'industryType=Telecommunications' \
  --form 'domain=Low Balance & Top-up' \
  --form 'file=@TMF629_Customer_Management_API_v4.0.0_swagger.json' \
  --form 'file=@TMF654_Prepay_Balance_Management_API_v4.0.0_swagger.json'
```

The server generates per-file source metadata internally. No per-file source id/name/standard/version fields are required in the request.

## LLM boundary

The LLM is provided only with:

- the user's current scenario/request text;
- the exact active MongoDB source catalog for the requested industry/domain, when present; and
- persisted MongoDB scenario-variable definitions, when present.

It is explicitly prevented from using static telecom registries, bundled standards files, external standards URLs, country/industry profiles, templates, examples, or generic domain knowledge as executable grounding. The deterministic compiler and validators enforce the same source boundary.

## Scenario generation

For an agentic scenario:

```text
industryType + domain
        ↓
MongoDB industry_source_documents
        ↓
source catalog
        ↓
LLM selection/review
        ↓
deterministic compilation
        ↓
HITL confirmation
        ↓
generation
```

When no source JSON exists but MongoDB `scenario_variables` exist, the executable schema is built directly from those persisted definitions and the LLM cannot add new executable fields.

When both are absent, generation is rejected.
