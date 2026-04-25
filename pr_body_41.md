## Summary
- **Problem**: `ProjectContext` is a simple DTO serialized with `joblib` that lacks versioning, type safety, and training metadata.
- **Need**: Refactor the context to include a schema version and additional metadata fields, ensuring backwards compatibility for legacy contexts.

## Changes
- Updated `ProjectContext` to include `schema_version`, `task_type`, `model_hash`, `training_date`, and `reference_stats`.
- Added `validate()` method to ensure schema integrity and `summary()` method to generate a DataFrame overview.
- Handled legacy context loading with a `DeprecationWarning` and automatic attribute backfilling.

## Test plan
- [x] **Local Validation**: Extensive unit tests covering initialization, serialization, summary, schema validation, and legacy loading.
- [x] **Security Check**: Handled using `joblib` securely without remote execution.
- [x] **CI Validation**: Passed `mypy --strict` and `make test`.

## Risk & rollback
- **Risk level**: low
- **Rollback**: Revert this PR.

## Related
- **Issue**: Closes #41
- **Milestone**: M1 - Core & Memory

## Checklist
- [x] Title follows `type(scope): short description`
- [x] Linked issues are referenced
- [x] Scope is small and focused
- [x] Tests/Docs updated (if applicable)
