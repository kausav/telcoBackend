# Edge-case contract

- `edgeCaseVariables` is optional.
- If it is empty/missing, `edgeCasePercentage` is normalized to `0`.
- `edgeCasePercentage` is a finite fraction in `[0, 1]`; `0.02` means 2%.
- Edge-case definitions are grouped by `edge_case_name` and may contain multiple variable overrides.
- Conditions are machine-checkable expressions over canonical field names.
- Transactional edge-case flags live inside `events[].records[]`.
- Aggregational edge-case flags live on each generated record.
- A record is marked `isEdgeCaseData=true` only after its condition evaluates to true.
- No variable-specific bypass is used. Conditions are converted into deterministic constraints for supported comparison/boolean expressions.
- Transactional sparse events only use fields declared by the event or explicitly declared by the edge-case definition; unrelated event fields are not borrowed.
- Formula fields remain authoritative and are recalculated after edge-case inputs are applied.
- If an edge-case condition is genuinely impossible under the confirmed schema, generation fails rather than silently labelling invalid data.
