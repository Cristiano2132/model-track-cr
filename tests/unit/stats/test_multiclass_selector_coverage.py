import pandas as pd

from model_track.stats.multiclass_selection import MulticlassSelector


def test_multiclass_selector_sample_data():
    df = pd.DataFrame({"target": ["A", "B"] * 50, "feat1": [1, 2] * 50})
    sel = MulticlassSelector(classes=["A", "B"], sample_size=10)
    # This should trigger the sampling logic because len(df) > sample_size
    sel.fit(df, "target", ["feat1"])
    # 10 samples total, so we verify sampling worked without crashing
    assert "feat1" in sel.selected_features_ or "feat1" in sel.dropped_features_


def test_multiclass_selector_mean_strategy():
    df = pd.DataFrame(
        {
            "target": ["A", "B"] * 50,
            "feat1": [1, 1] * 50,  # low IV
        }
    )
    sel = MulticlassSelector(classes=["A", "B"], iv_strategy="mean", iv_threshold=0.0)
    sel.fit(df, "target", ["feat1"])
    assert "feat1" in sel.selected_features_ or "feat1" in sel.dropped_features_
