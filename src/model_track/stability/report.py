from typing import Any

import pandas as pd

from ..base import TaskType
from ..context import ProjectContext
from .psi import ModelPSI, MulticlassPSI, PSICalculator, RegressionPSI


class StabilityReport:
    """
    Orchestrator for data stability and model drift monitoring.
    Combines feature-level PSI and score-level PSI into a unified report.
    """

    def __init__(
        self,
        context: ProjectContext | None = None,
        feature_threshold: float = 0.25,
        score_threshold: float = 0.10,
    ):
        self.context = context
        self.feature_threshold = feature_threshold
        self.score_threshold = score_threshold
        self.feature_psi_ = PSICalculator()
        self.score_psi_ = ModelPSI()
        self.multiclass_psi_ = MulticlassPSI()
        self.regression_psi_ = RegressionPSI()
        self.results_: dict[str, Any] = {}

        if context:
            # Load reference stats from context if available
            self.feature_psi_ = PSICalculator.from_context(context)

    @classmethod
    def from_context(
        cls,
        context: ProjectContext,
        feature_threshold: float = 0.25,
        score_threshold: float = 0.10,
    ) -> "StabilityReport":
        """Factory method to build report from context."""
        return cls(
            context=context,
            feature_threshold=feature_threshold,
            score_threshold=score_threshold,
        )

    def run(
        self,
        df: pd.DataFrame,
        features: list[str] | None = None,
        score_col: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Execute stability checks for features and scores.
        """
        self._prepare_score_col(score_col)
        report_data = []

        # 1. Feature PSI
        feat_list = self._get_feature_list(features, score_col)
        if feat_list:
            report_data.extend(self._process_feature_psi(df, feat_list))

        # 2. Score PSI
        if score_col:
            report_data.extend(self._process_score_psi(df, score_col))

        self.results_["data"] = pd.DataFrame(report_data)
        return self.results_["data"]

    def _prepare_score_col(self, score_col: str | list[str] | None) -> None:
        """Cleanup score_col from feature_psi reference stats if needed."""
        if score_col:
            cols = [score_col] if isinstance(score_col, str) else score_col
            for col in cols:
                if col in self.feature_psi_.reference_stats_:
                    del self.feature_psi_.reference_stats_[col]

    def _get_feature_list(
        self, features: list[str] | None, score_col: str | list[str] | None
    ) -> list[str] | None:
        """Determine which features to analyze."""
        if features is not None:
            return features
        if self.context:
            cols_to_exclude = [score_col] if isinstance(score_col, str) else (score_col or [])
            all_keys = list(getattr(self.context, "reference_stats", {}).keys())
            return [k for k in all_keys if k not in cols_to_exclude]
        return None

    def _process_feature_psi(self, df: pd.DataFrame, feat_list: list[str]) -> list[dict[str, Any]]:
        """Calculate PSI for features and format results."""
        report_data = []
        feat_summary = self.feature_psi_.transform(df)
        # Filter only requested features
        for _, row in feat_summary.iterrows():
            if row["feature"] in feat_list:
                report_data.append(
                    {
                        "type": "feature",
                        "name": row["feature"],
                        "psi": row["psi"],
                        "status": row["status"],
                    }
                )
        return report_data

    def _process_score_psi(
        self, df: pd.DataFrame, score_col: str | list[str]
    ) -> list[dict[str, Any]]:
        """Calculate PSI for scores and format results."""
        report_data = []

        if isinstance(score_col, list):
            self.multiclass_psi_.proba_cols_ = score_col
            if not self.multiclass_psi_.reference_stats_ and self.context:
                ctx_stats = getattr(self.context, "reference_stats", {})
                ref_stats = {}
                for col in score_col:
                    if col in ctx_stats:
                        ref_stats[col] = ctx_stats[col]
                self.multiclass_psi_.reference_stats_ = ref_stats

            try:
                score_summary = self.multiclass_psi_.transform(df)
                for _, row in score_summary.iterrows():
                    report_data.append(
                        {
                            "type": "score",
                            "name": row["feature"],
                            "psi": row["psi"],
                            "status": row["status"],
                        }
                    )
            except (ValueError, KeyError):
                pass
        else:
            is_regression = self.context and self.context.task_type == TaskType.REGRESSION
            calc = self.regression_psi_ if is_regression else self.score_psi_

            calc.score_col_ = score_col
            if not calc.reference_stats_ and self.context:
                ctx_stats = getattr(self.context, "reference_stats", {})
                if score_col in ctx_stats:
                    calc.reference_stats_ = {score_col: ctx_stats[score_col]}

            try:
                score_summary = calc.transform(df)
                for _, row in score_summary.iterrows():
                    report_data.append(
                        {
                            "type": "score",
                            "name": row["feature"],
                            "psi": row["psi"],
                            "status": row["status"],
                        }
                    )
            except (ValueError, KeyError):
                pass
        return report_data

    def generate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Alias for run() to match Issue #50 specs."""
        return self.run(*args, **kwargs)

    def summary(self) -> dict[str, Any]:
        """Returns executive summary of the stability report."""
        if "data" not in self.results_ or self.results_["data"].empty:
            return {"status": "Not Run", "unstable_count": 0}

        df = self.results_["data"]
        unstable_df = df[df["status"] == "Unstable"]
        monitor_df = df[df["status"] == "Monitor"]

        status = "Stable"
        if not unstable_df.empty:
            status = "Unstable"
        elif not monitor_df.empty:
            status = "Monitor"

        return {
            "overall_status": status,
            "unstable_features": unstable_df[unstable_df["type"] == "feature"]["name"].tolist(),
            "unstable_scores": unstable_df[unstable_df["type"] == "score"]["name"].tolist(),
            "metrics": {
                "total_items": len(df),
                "unstable_count": len(unstable_df),
                "monitor_count": len(monitor_df),
            },
        }

    def summary_text(self) -> str:
        """Returns a human-readable string summary."""
        res = self.summary()
        if res.get("status") == "Not Run":
            return "Stability Report: NOT RUN"

        text = f"Stability Report: {res['overall_status'].upper()}\n"
        text += f"- Total monitored items: {res['metrics']['total_items']}\n"
        text += f"- Unstable items: {res['metrics']['unstable_count']}\n"

        if res["unstable_features"]:
            text += f"- Unstable Features: {', '.join(res['unstable_features'])}\n"
        if res["unstable_scores"]:
            text += f"- Unstable Scores: {', '.join(res['unstable_scores'])}\n"

        return text

    def is_healthy(self) -> bool:
        """Returns True if overall status is Stable."""
        res = self.summary()
        return res.get("overall_status") == "Stable"

    def plot_drift_heatmap(self, ax: Any = None) -> Any:
        """Generates a color-coded PSI status heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "Seaborn/Matplotlib required for plotting. Install with [viz] extra."
            ) from None

        if "data" not in self.results_ or self.results_["data"].empty:
            raise ValueError("Run the report before plotting.")

        df = self.results_["data"].copy()
        # Map status to numeric for heatmap
        status_map = {"Stable": 0, "Monitor": 1, "Unstable": 2}
        df["status_val"] = df["status"].map(status_map)

        # Reshape for heatmap: rows=features, cols=psi (simplified for single run)
        plot_df = df.set_index("name")[["status_val"]]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, len(plot_df) * 0.4 + 2))

        sns.heatmap(
            plot_df,
            annot=df.set_index("name")[["psi"]],
            fmt=".3f",
            cmap="RdYlGn_r",
            cbar=False,
            ax=ax,
            linewidths=0.5,
            vmin=0,
            vmax=2,
        )
        ax.set_title("Stability Drift Heatmap (PSI)")
        ax.set_xlabel("Monitoring Cycle")
        ax.set_ylabel("Item")
        return ax
