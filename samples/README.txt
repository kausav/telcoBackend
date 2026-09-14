CSV scenario sample
===================

Use `definition_sample.csv` as the single canonical CSV example for both transactional and aggregational scenarios. The API `typeOfData` form field determines which generation mode is used; the CSV format stays the same.

Schema columns
- `scope`: use `user` for stable user/entity attributes in transactional history generation; use `record` for fields generated on each historical row. Aggregational generation treats the schema as one flat record definition.
- `name`: variable name.
- `dtype`: variable datatype.
- `description`: business meaning of the variable.
- `gen`: generator type.
- `params`: generator parameters, either JSON or compact `key=value;key=value` syntax. Bare values are also supported for simple generators.
- `depends_on`: comma-separated variable dependencies.
- `nullable`: TRUE/FALSE.
- `formula`: optional executable formula.

The sample intentionally includes both `weighted_choice` and `boolean` generators.

Transactional output
- `count` means the number of users/entities.
- `recordsPerUser` controls the number of recent history rows returned per user and defaults to 10.
- User-scope fields are emitted once per user; record-scope fields appear inside that user’s `records` array.

Aggregational output
- `count` means the number of flat records returned.

There is no event model in the current API or CSV schema: no `event_type`, event sequence, event grouping, `events`, or event-specific confirm operations.
