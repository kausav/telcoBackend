CSV scenario-definition format

record_type=variable: normal variable definition.
record_type=edge_case_variable: override/condition for an existing normal variable. The same variable may appear in multiple edge cases.
record_type=event: transactional event definition.
record_type=metadata: optional edge_case_percentage in [0,1].

For edge cases, edge_case_name, edge_case_description, and condition are required.
For edge_case_percentage, 0.02 means 2%. If no edge-case variables exist, the backend normalizes the percentage to 0.
