## Summary
- **Problem**: Need to evaluate risk models through fraud capture tables based on percentage thresholds.
- **Need**: Extract the custom capture table from the EDA baseline notebook into a standardized, tested `DecisionTable` evaluator component.

## Changes
- Implemented `DecisionTable` in `src/model_track/evaluation/decision_table.py` with `generate()`, `cutoff_for_capture()`, `decline_rate_for_capture()`, and `plot()`.
- Exposed `DecisionTable` from `evaluation/__init__.py`
- Added comprehensive unit tests and a notebook integration test ensuring parity.

## Test plan
- [x] **Local Validation**: 100% unit test coverage for `decision_table.py` + hypothesis edge cases
- [x] **Security Check**: No external data calls.
- [x] **CI Validation**: Passed `make test` locally.

## Risk & rollback
- **Risk level**: low
- **Rollback**: Revert this PR.

## Related
- **Issue**: Closes #44
- **Milestone**: M2 - Evaluation

## Checklist
- [x] Title follows `type(scope): short description`
- [x] Linked issues are referenced
- [x] Scope is small and focused
- [x] Tests/Docs updated (if applicable)
