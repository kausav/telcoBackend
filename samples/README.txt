Canonical CSV schema sample
===========================

Use `definition_sample.csv` as the single schema-definition sample for both
transactional and aggregational data. The API `typeOfData` field selects the
generation mode. The CSV itself does not contain a `scope` column and has no
event definitions.

Required columns:
- `name`
- `dtype`
- `gen`

Optional columns:
- `description`
- `params`
- `depends_on`
- `nullable`
- `formula`

For transactional generation, the backend identifies the entity with the API
`entityKey` and infers the user/history split from the first history timestamp
field. Fields before that timestamp form stable user/entity context; fields from
the timestamp onward form repeated history records. Aggregational generation
treats all variables as one flat record definition.

The sample demonstrates weighted choices and boolean generation.
