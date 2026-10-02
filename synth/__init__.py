"""Spec-driven synthetic data generation.

Layers (each independently testable, none depends on MongoDB or a model at import time):

    definitions - execute a variable's own definition (dtype, generator, params, formula)
    baseline    - a runnable spec from definitions alone (scope, clock, temporal order)
    compiler    - a language model designs the scenario's behaviour over the baseline; the simulator verifies it
    spec        - the validated, data-only description of that behaviour (no code, no per-industry data)
    engines     - simulate entity timelines from a spec (consistency by construction)
    contract    - the output contract implied by the variable definitions themselves
    projection  - write simulated values into the confirmed columns
    scorer      - measure the delivered rows (invariants, targets, contract, structure)
    service     - ensure_spec / generate; store keeps verified specs
"""
