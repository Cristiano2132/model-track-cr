import pandas as pd

from model_track.binning import BinApplier, QuantileBinner, TreeBinner
from model_track.context import ProjectContext


def test_binning_workflow_with_context():
    # 1. Setup data
    df_train = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45, 50],
            "income": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
            "target": [0, 0, 0, 1, 1, 1, 1],
        }
    )

    # 2. Fit binners
    tree_binner = TreeBinner(max_depth=2, min_samples_leaf=1)
    tree_binner.fit(df_train, column="age", target="target")

    quant_binner = QuantileBinner(n_bins=2)
    quant_binner.fit(df_train, column="income")

    # 3. Save to context
    ctx = ProjectContext()
    ctx.bins_map["age"] = tree_binner.bins
    ctx.bins_map["income"] = quant_binner.bins

    # 4. Use BinApplier on new data
    df_test = pd.DataFrame({"age": [22, 48], "income": [1500, 6500]})

    applier = BinApplier.from_context(ctx)
    df_transformed = applier.apply(df_test)

    # 5. Verify results
    assert df_transformed["age"].nunique() == 2
    assert df_transformed["income"].nunique() == 2

    # Check specific values
    # For age, tree split should be around 32.5
    # For income, median should be 4000
    assert df_transformed["age"].iloc[0] != df_transformed["age"].iloc[1]
    assert df_transformed["income"].iloc[0] != df_transformed["income"].iloc[1]

    # Ensure they are strings
    assert isinstance(df_transformed["age"].iloc[0], str)
