"""ER-model proof of concept (branch ``feature/er-model-poc``).

Several related tables generated from one entity-relationship description, with referential integrity, cardinality,
business constraints and an independent integrity report. Pure standard library: no database, no model call.

    model.py       parse and validate the ER JSON
    generators.py  the small closed catalogue of column generators
    engine.py      parent-first generation, relationships, constraints
    integrity.py   an independent checker that re-reads the generated tables
    export.py      CSV and zip output
    handlers.py    the API behaviour (plain functions: request data in, status and payload out)
    api.py         FastAPI router (mounted in server.py on this branch)
"""
VERSION = "0.1.0-poc"
