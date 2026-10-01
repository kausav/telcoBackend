"""Behaviour-driven synthetic data generation.

Layers (each independently testable, none depends on an LLM or on MongoDB at import time):

    concepts   - map source-model fields (TMF JSON, DB variables) onto business *concepts*,
                 collapse duplicates, prune out-of-scope resources
    pack       - versioned, data-only description of a domain's behaviour (vocabularies,
                 distributions, invariants, distribution targets)
    engines    - simulate entity journeys in concept space (consistency by construction)
    projection - write concept values into the confirmed columns
    scorer     - measure accuracy (invariants + distribution fidelity + structure)
"""
