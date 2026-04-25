import os
import tempfile
from datetime import datetime

import joblib
import pandas as pd
import pytest

from model_track.base import TaskType
from model_track.context import ProjectContext, VersionMismatchError


class TestProjectContext:
    def test_context_initialization(self) -> None:
        ctx = ProjectContext()
        assert ctx.schema_version == "2.0"
        assert ctx.task_type == TaskType.BINARY
        assert ctx.target is None
        assert ctx.metadata == {}
        assert ctx.bins_map == {}
        assert ctx.woe_maps == {}
        assert ctx.selected_features == []
        assert ctx.model_hash is None
        assert ctx.training_date is None
        assert ctx.reference_stats is None

    def test_summary_returns_dataframe(self) -> None:
        ctx = ProjectContext()
        ctx.training_date = datetime(2023, 1, 1, 12, 0)
        df = ctx.summary()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Field", "Value"]
        # Schema version check
        assert df.loc[df["Field"] == "Schema Version", "Value"].values[0] == "2.0"
        # Training date format check
        assert df.loc[df["Field"] == "Training Date", "Value"].values[0] == "2023-01-01T12:00:00"

    def test_save_and_load(self) -> None:
        ctx = ProjectContext()
        ctx.task_type = TaskType.REGRESSION
        ctx.model_hash = "abc123hash"
        ctx.metadata["test_key"] = "test_value"
        ctx.training_date = datetime(2024, 1, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "context.joblib")
            ctx.save(path)

            loaded_ctx = ProjectContext.load(path)

            assert loaded_ctx.schema_version == "2.0"
            assert loaded_ctx.task_type == TaskType.REGRESSION
            assert loaded_ctx.model_hash == "abc123hash"
            assert loaded_ctx.metadata["test_key"] == "test_value"
            assert loaded_ctx.training_date == datetime(2024, 1, 1)

    def test_validate_raises_on_invalid_version(self) -> None:
        ctx = ProjectContext()
        ctx.schema_version = "1.0"
        with pytest.raises(VersionMismatchError, match="Incompatible schema version"):
            ctx.validate()

    def test_legacy_context_loading_with_warning(self) -> None:
        legacy_ctx = ProjectContext()
        # Simulate legacy state
        del legacy_ctx.schema_version
        del legacy_ctx.task_type
        del legacy_ctx.model_hash
        del legacy_ctx.training_date
        del legacy_ctx.reference_stats
        legacy_ctx.target = "y"
        legacy_ctx.selected_features = ["a"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.joblib")
            joblib.dump(legacy_ctx, path)

            with pytest.warns(DeprecationWarning, match="Loading legacy context"):
                loaded_ctx = ProjectContext.load(path)

            # Verify backfill
            assert loaded_ctx.schema_version == "2.0"
            assert loaded_ctx.task_type == TaskType.BINARY
            assert loaded_ctx.model_hash is None
            assert loaded_ctx.training_date is None
            assert loaded_ctx.reference_stats is None
            assert loaded_ctx.target == "y"
            assert loaded_ctx.selected_features == ["a"]
