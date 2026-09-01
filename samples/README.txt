CSV scenario-definition samples (UPLOAD/INPUT examples)

These files are scenario-definition CSVs, NOT final generated data. They describe what the user wants the generator to produce.

transactional_definition_sample.csv
- Normal variables describe the transactional data model.
- Event rows define which variables belong to each event.
- edge_case_variable rows define edge-case overrides and their machine-checkable condition.
- The transactional sample includes two edge cases and edgeCasePercentage=0.02 (2%).

aggregational_definition_sample.csv
- Normal variables describe one aggregational record.
- edge_case_variable rows define edge-case overrides and their machine-checkable condition.
- The aggregational sample includes three edge cases and edgeCasePercentage=0.04 (4%).

CSV rules
- record_type=variable: normal variable definition.
- record_type=edge_case_variable: edge-case override/condition. Multiple rows may belong to the same edge_case_name.
- record_type=event: transactional event definition.
- record_type=metadata with name=edgeCasePercentage and value=<number>: optional edge-case percentage.
- edgeCasePercentage must be a finite number between 0 and 1.
- If no edge-case variables are supplied, the backend normalizes edgeCasePercentage to 0.
- Conditions must reference declared variables and use supported expression syntax.
