# ER model proof of concept

Branch: `feature/er-model-poc` (baseline: `main`, API 2.14.2). This is a testing-the-waters build: it generates **several related
tables from one entity-relationship (ER) description**, with referential integrity, relationship counts, business rules and an
independent integrity report. It uses only the Python standard library: no MongoDB, no Gemini call, no new dependency.
The existing `/scenario/*` flow is untouched.

## What is in the branch

| Path | What it is |
|---|---|
| `er/model.py` | Reads and checks the ER JSON; every problem is reported at once, with the path of the value |
| `er/generators.py` | The nine column generators and the seeded random streams |
| `er/engine.py` | Parent-first generation, relationships, business rules, totals |
| `er/integrity.py` | A separate checker that re-reads the delivered tables and verifies every promise |
| `er/export.py` | CSV and zip output |
| `er/handlers.py` | API behaviour as plain functions, including `failure()`, the error envelope |
| `er/api.py` | FastAPI router, mounted in `server.py` (6 added lines, a version label `2.14.2+er-poc`, and `expose_headers` in the CORS settings) |
| `er/samples/` | `telecom_family.json`, `banking_holders.json`, `retail_snowflake.json` (a snowflake: one fact table, dimensions normalised into sub-dimensions) |

## Run it

```bash
uvicorn server:app --port 8000       # routes: /er/samples, /er/validate, /er/generate, /er/export (behind the API token, like every route)
```

The screen that uses these routes is in the frontend repository, branch `feature/er-model-poc` (see "The real frontend" below). There is no
separate sample page and no test folder in this branch.

## The ER model format

```jsonc
{
  "name": "…", "description": "…", "seed": 42,                     // seed: 0 to 2147483647
  "entities": [ { "name": "customer", "key": "customer_id", "count": 40, "columns": [ {"name": "…", "gen": {…}} ] } ],
  "relationships": [ { "name": "customer_accounts", "type": "one_to_many", … } ],
  "constraints":   [ { "type": "date_not_before_parent", "relationship": "customer_accounts", … } ]
}
```

**Entities.** `name`, `key` (default `<name>_id`), `count` (needed only for a root entity, one whose rows no relationship creates),
`columns` in the order they should be delivered. The key must use the `id` generator; if it is not declared it is added.
Delivered column order is always **key, foreign keys, then the declared columns**. A foreign-key column is added to the child
automatically; declaring one is allowed but its generator is ignored (a warning says so).

**Column generators** (`gen.type`; any generator also takes `null_share` 0 to 1):

| Type | Parameters |
|---|---|
| `id` | `prefix`, `digits` (6), `start` (1): sequential and unique |
| `choice` | `weights` (`{"label": weight}`) or `values` (a list) |
| `int` / `float` | `min`, `max` (and `round` for float, 2) |
| `date` | `start`, `end` (ISO dates) |
| `bool` | `p` (0.5) |
| `const` | `value` |
| `date_after` | `column` (an earlier date column), `min_days` (0), `max_days` (365): for "end date after start date" |
| `digits` | `length`, `prefix`, `unique`: phone-number-like digits |

**Relationships** (`type`):

| Type | Fields | Meaning |
|---|---|---|
| `one_to_many` | `parent`, `child`, `fk`, `children` `{min, max, weights?}` | Creates the child rows: each parent gets min to max children |
| `many_to_one` | `parent`, `child`, `fk`, `null_share`, `skew` | A lookup: each child row points at one parent; `skew` makes some parents hubs |
| `many_to_many` | `left`, `right`, `through`, `left_fk`, `right_fk`, `per_right` `{min, max, weights?}`, `cover_left` | Creates the rows of the `through` (junction) table: each right row is linked to min to max distinct left rows, never the same pair twice; `cover_left` links every left row at least once when there are enough pairs |
| `group` | `entity`, `fk`, `share`, `members` `{min, max}` | Rows of one table grouped under a main row of the same table (a main account with member accounts). The main is always listed before its members, and a main is never a member |

An entity's row count comes from its `count` or from exactly one `one_to_many` / `many_to_many`; anything else is reported.
A loop between entities is reported by name.

**Constraints** (`type`, each tied to a relationship by `relationship`):

| Type | Fields | Effect |
|---|---|---|
| `date_not_before_parent` | `column`, `parent_column`, `max_days_after` (365) | A child date earlier than its parent's is moved to the parent's date plus 0 to max days; dates measured with `date_after` from it follow |
| `same_as_parent` | `column`, `parent_column` | Copied from the parent |
| `status_follows_parent` | `column`, `parent_column`, `when`, `set` | When the parent value is `when` (a value or list), the child becomes `set` |
| `aggregate` | `parent_column`, `child_column`, `fn` (`sum`, `count`, `min`, `max`, `avg`) | A parent total computed from its children, so it can never disagree with them |

Limits: 30 entities, 80 columns each, 5000 rows for a root entity, 500 children per parent, 200,000 rows in a run (checked
before generating). Measured here: about 82,000 rows generate in 0.7 s and are checked in 0.2 s. The same model and seed always
give identical tables; every entity has its own random stream, so adding a rule never reshuffles unrelated data.

## API contract

All bodies are JSON. The body of `validate`, `generate` and `export` is
`{"model": {…}, "seed": 42 (optional), "counts": {"customer": 100} (optional, root entities only)}`.

| Method and path | Returns |
|---|---|
| `GET /er/samples` | `{samples: [{id, name, description, entities, relationships}]}` |
| `GET /er/samples/{id}` | `{id, model}`; 404 if unknown |
| `POST /er/validate` | **200** always: `{valid, errors[], warnings[], plan}`; `plan` is null when invalid |
| `POST /er/generate` | **200** `{seed, order[], total_records, elapsed_ms, warnings[], tables{}, integrity_report}`; **422** an error (below) |
| `POST /er/export` | **200** zip: one `<table>.csv` each, `integrity_report.json`, `model.json`; **422** as above |

**Errors** use the application's own envelope (`core/errors.py`), so the front end's one error reader works on them:

```json
{"success": false,
 "error": {"code": "VALIDATION_ERROR", "message": "The ER model has 2 problems",
           "details": [{"field": "entities[1].columns[2].gen.type", "message": "unknown generator 'magic'; one of [...]"},
                       {"field": null, "message": "entity 'a' has no parent relationship creating its rows, so it needs a 'count'"}],
           "request_id": "…"},
 "errors": ["…the same sentences, flat…"], "warnings": ["…"]}
```

`code` is `VALIDATION_ERROR` (422) or `NOT_FOUND` (an unknown sample, 404). `field` is the path of the value when the sentence starts with
one. The flat `errors` and `warnings` lists stay beside it; the front end's ER screen reads them for a rejected generate. `request_id` is the request's own id
(also the `X-Request-ID` header). CORS now exposes `Content-Disposition` and `X-Request-ID` so a browser can read the zip's file name.

`plan` (what a review screen shows): `order[]` (tables parent-first), `entities[]` (`name`, `key`, `role` root / child / junction,
`created_by`, `rows_expected`, `columns[]` with `kind` key / fk / column, `generator`, or `references` and `relationship`),
`relationships[]`, `constraints[]`, `projected_rows`, `limits`.

`tables{}` is keyed by table name, in creation order:

```json
"account": {
  "fields": ["account_id", "customer_id", "opened"],
  "primary_key": "account_id",
  "foreign_keys": [{"column": "customer_id", "references": "customer.customer_id", "relationship": "owns"}],
  "total_records": 3,
  "records": [{"account_id": "A001", "customer_id": "C001", "opened": "2024-10-23"}]
}
```

`integrity_report` is `{passed, summary{checks, passed, failed, warnings, info, tables, total_rows}, checks[]}`; a check is
`{id, kind (table | relationship | constraint | column), scope, description, status (pass | fail | warn | info), expected, observed}`
(constraint checks also carry `rows_checked`). Example:
`{"id": "fk.owns.orphans", "status": "pass", "expected": 0, "observed": 0, "description": "every account.customer_id points at an existing customer"}`.
The report passes when no check failed.

Errors are plain sentences with the path of the value, for example
`entities[1].columns[2].gen.type: unknown generator 'magic'; one of [...]` or
`entity 'a' has no parent relationship creating its rows, so it needs a 'count'`.

## What the checker verifies

Per table: delivered columns and order, unique and non-empty keys, declared value ranges and lists. Per relationship: no orphan
foreign keys, none empty where required, children per parent inside the declared range (and weights), many-to-many pairs unique,
links per right row in range, left coverage, group size, no loops, no self-parent, depth of one, main listed before members.
Per rule: each business rule on every row. Per `date_after` column: the gap in days.

## How this answers the requirements note

| Requirement | Where |
|---|---|
| R1 several related tables | one table per entity, delivered under `tables` and as one CSV each |
| R2 parents first, zero orphans | `plan.order`; orphan checks in the report |
| R3 foreign-key mapping | automatic FK columns with the parent key's format; `foreign_keys` in each table |
| R4 kinds of links and counts | `one_to_many`, `many_to_one`, `many_to_many`, `group`, each with min, max and weights |
| R5 business rules | the four constraint types, checked row by row |
| R6 the ER model as input | the JSON above; the page edits it and draws it |
| R7 proof | `integrity_report`, written by a checker independent of the generator |

## The real frontend

The frontend repository's branch `feature/er-model-poc` (from its `master`, the delivered app) adds an **ER model** screen that uses these
routes through the app's own axios client (Bearer token from `VITE_API_TOKEN`, base URL `VITE_API_HOST`). The changes made on this side
for it: the error envelope and CORS `expose_headers` above. Nothing else in the API had to change.

## For the UI developer

The frontend branch is the reference for how the contract is used: model editor with live validation, an ER diagram drawn from `plan`, the plan in
plain words, tabbed data with foreign-key values that open the parent row, the integrity report, and CSV or zip download. All calls are in its
`src/api/erApi.ts`; everything else reads `plan` and `tables`. The screen shows the first 200 rows of a table and filters across all rows; a
production screen should page or virtualise. Values come from user models and generated data, so show them as text, never as markup.

## Not built yet (deliberately, for a first test)

- The model proposing the ER model from a scenario (no Gemini in this branch), and the CSV-pair or SQL DDL imports.
- Rules on a many-to-many relationship, and groups deeper than one level (a main with members, not members of members).
- Integration with the `/scenario/propose → confirm → generate` flow, the variable library and MongoDB storage.
- History over time (a record changing plan) and a joined single-table export.
- Authentication of its own: the application's token middleware covers the `/er/*` routes.

## Verified here, and not

Verified in this workspace, with unit tests that are not part of this branch: the model checks, the generators, the engine (same seed, same
data), the independent integrity checker (21 deliberate corruptions, all caught), the handlers and the error envelope. The router was run only
through stand-in classes, and the real frontend screen was run against these handlers over HTTP in a headless browser.
**Not run here:** `server.py` with the real FastAPI router. FastAPI could not be installed in this workspace, so `er/api.py` and `server.py`
were compiled but never started. The router only forwards to `er/handlers.py`; start the application once and open `/docs` to confirm the
five `/er/*` routes appear.

## Moving it to your repository

This directory is a git repository with `main` (the delivered 2.14.2) and `feature/er-model-poc`. To bring it into your own
repository, either copy the branch, or apply the patch from the folder you have checked out:

```bash
git checkout -b feature/er-model-poc
git apply er-model-poc.patch          # adds er/ and docs/, and a few lines in server.py (router, version label, CORS expose_headers)
```
