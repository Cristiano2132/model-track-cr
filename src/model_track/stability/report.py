from typing import Any

import pandas as pd

from ..context import ProjectContext
from .psi import ModelPSI, PSICalculator


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
        score_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Execute stability checks for features and scores.
        """
        # Cleanup score_col from feature_psi reference stats if it was loaded from context
        if score_col and score_col in self.feature_psi_.reference_stats_:
            del self.feature_psi_.reference_stats_[score_col]

        report_data = []

        # 1. Feature PSI
        feat_list = features
        if feat_list is None and self.context:
            # Try to get features from context reference_stats keys
            all_keys = list(getattr(self.context, "reference_stats", {}).keys())
            feat_list = [k for k in all_keys if k != score_col]

        if feat_list:
            feat_summary = self.feature_psi_.transform(df)
            for _, row in feat_summary.iterrows():
                report_data.append(
                    {
                        "type": "feature",
                        "name": row["feature"],
                        "psi": row["psi"],
                        "status": row["status"],
                    }
                )

        # 2. Score PSI
        if score_col:
            self.score_psi_.score_col_ = score_col
            if not self.score_psi_.reference_stats_ and self.context:
                ctx_stats = getattr(self.context, "reference_stats", {})
                if score_col in ctx_stats:
                    self.score_psi_.reference_stats_ = {score_col: ctx_stats[score_col]}

            try:
                score_summary = self.score_psi_.transform(df)
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
                # Handle cases where score column might not be fitted or missing in DF
                pass

        self.results_["data"] = pd.DataFrame(report_data)
        return self.results_["data"]

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
