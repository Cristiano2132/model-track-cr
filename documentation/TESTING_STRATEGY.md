# Testing Strategy and Statistical Validation (v0.5.0)

*Read this in other languages: [English](TESTING_STRATEGY.md), [Português](TESTING_STRATEGY.pt-br.md)*

## 1. Testing Philosophy
The `model_track_cr` project adopts a **"Defense in Depth"** approach. We do not merely validate whether the code "runs without crashing"—we rigorously ensure that the statistical output is mathematically consistent, robust to noisy data, and computationally efficient.

<p align="center">
  <img src="images/pyramid.png" alt="Testing Pyramid for Data Science" width="600">
</p>

*Our pyramid prioritizes a strong foundation of unit tests, reinforced by automated statistical validations and property-based assertions.*

## 2. Testing Structure (Mirror Strategy)
To maintain scalability and a clear cognitive mapping, we adopt the **Mirror Strategy**, where the testing directories reflect the exact structure of `src/model_track`.

- **`tests/unit/`**: Black-box tests for individual functions and methods. Focused on line coverage and exception handling.
- **`tests/statistical/`**: Property-Based Tests (PBT). Validates mathematical invariances using synthetic data generation.
- **`tests/integration/`**: End-to-end workflow tests (e.g., from raw DataFrame ingestion to the final correlation report or scorecard).
- **`tests/benchmarks/`**: Performance monitoring and algorithmic complexity (Big O) evaluations.

## 3. Property-Based Testing (PBT)
We utilize the `Hypothesis` library to challenge our implementations with data structures and edge cases that a human would rarely consider encoding manually.

### Generation Strategies:
- **Extreme DataFrames:** Generating columns with 100% NaNs, constant values, and floating-point values near precision limits (e.g., `-1e6` to `1e6`).
- **Type Robustness:** Testing mixtures of types (`float`, `int`, `category`) to ensure that classes (like `Analyzer` or `TypeDetector`) either fail gracefully or convert types correctly.

## 4. Invariance & Statistical Sanity
Every implemented metric must pass strict sanity tests:
- **Symmetry:** $Corr(X, Y) == Corr(Y, X)$.
- **Mathematical Bounds:** Ensure that correlations are strictly within the $[-1, 1]$ interval, and metrics like PSI or KS are always $\ge 0$.
- **Scale Invariance:** Multiplying a column by a constant should not alter the Pearson correlation coefficient or the Information Value (IV).

## 5. Coverage and Quality
- **Coverage Goal:** Minimum of 95% coverage across core modules (currently operating at **100%**).
- **Error Handling:** Failure paths (e.g., arrays of mismatched shapes, zero variance columns) must be explicitly tested using `pytest.raises`.

## 6. Performance Benchmarking (Big O)
To prevent performance regressions on large volumes of data:
- **Baselines:** Execution time measurements for $N=10^3, 10^5, 10^6$ rows.
- **Target Complexity:** Single-column operations should tend to $O(n)$. Correlation matrices and stability calculations must be optimized via NumPy/Pandas vectorization to avoid pure Python $O(n^2)$ bottlenecks.

## 7. Compatibility Matrix
Our CI/CD pipeline validates the library against:
- **Python:** 3.10, 3.11, 3.12, 3.13.
- **Pandas:** v1.x and v2.x (including support for the PyArrow backend).
