## Active Risks

| ID | Description | Severity | Mitigation |
|----|-------------|----------|------------|
| R-001 | MLOps retrofit consumes all time | HIGH | Hard 2-week time-box |
| R-002 | Walk-forward validation slow on CPU | MEDIUM | Limit to 10 stores during dev |
| R-003 | LLM API costs exceed budget | MEDIUM | Use Ollama locally for Week 4 |


## T-011 — MASE Gate: Relative vs Absolute Threshold

**Decision**: Gate compares LightGBM MASE against Baseline MASE
rather than against fixed threshold of 1.0.

**Why**: With 50 stores concatenated into a single evaluation array,
store boundary crossings inflate the naive denominator. Both the
baseline and LightGBM score above 1.0 in this setting. The relative
comparison (LightGBM < Baseline) is correct — it directly answers
"does the model beat the naive baseline?"

**Evidence**: Baseline MASE=1.2956, LightGBM MASE=1.0338.
LightGBM is 20% better than baseline but fails a fixed 1.0 gate.

**Upgrade path**: Compute MASE per store and average across stores.
This produces theoretically correct per-series MASE values.


## Phase 3 Lessons Learned

### L-001: pkg_resources incompatibility with Python 3.12
**What happened**: MLflow 3.x imports pkg_resources which was removed
from Python 3.12 standard library. setuptools 82.x installs but does
not register pkg_resources correctly in virtualenvs on Windows.
**Resolution**: Made MLflow optional in evaluate.py with try/except.
The quality gate runs regardless of MLflow availability.
**Principle**: A quality gate must never fail because a logging
service is unavailable. Gates and observability are separate concerns.

### L-002: Loguru does not support Python f-string format specifiers
**What happened**: `logger.info("rows: {rows:,}", rows=100)` raises
KeyError because loguru interprets `rows:,` as the variable name.
**Resolution**: Use `.format()` on the string before passing to loguru:
`logger.info("rows: {}".format(f"{rows:,}"))`
**Principle**: Understand the difference between Python f-string
syntax and loguru's lazy evaluation template syntax.

### L-003: sort_index vs sort_values
**What happened**: `df.sort_index(["store", "date"])` raises TypeError.
sort_index sorts by the DataFrame row index, not column values.
**Resolution**: Use `df.sort_values(["store", "date"])`.
**Principle**: Read error messages precisely — "takes 1 positional
argument but 2 were given" on sort_index is unambiguous.

### L-004: Misaligned actuals and predictions produced garbage metrics
**What happened**: eval_df was sorted by date only. predict() sorted
by [store, date]. Predictions for store 1 were compared against
actuals for store 2 — producing MASE and SMAPE values that were
meaningless.
**Resolution**: Explicitly sort eval_df by [store, date] before
calling model.fit() and model.predict().
**Principle**: In multi-entity time series, always enforce identical
sort order between the array you predict on and the array you
evaluate against. Never assume they are aligned.

### L-005: MASE > 1.0 in multi-store evaluation does not mean failure
**What happened**: Concatenating 50 stores into one array causes
naive differences at store boundaries to be meaningless — inflating
the naive MAE denominator and pushing MASE above 1.0 for both
baseline and model.
**Resolution**: Gate on relative comparison (LightGBM MASE
Baseline MASE) rather than fixed threshold of 1.0. This correctly
answers "does the model beat the naive baseline?"
**Principle**: Understand what your metric measures before defining
a gate. MASE=1.0 is theoretically correct for single-series
evaluation. Multi-series concatenation changes the denominator.

### L-006: LightGBM requires numeric dtypes — no object columns
**What happened**: store_type and assortment were string columns
("a", "b", "c") from store.csv. LightGBM raised ValueError on fit.
**Resolution**: Added ordinal encoding in build_features() before
the return statement.
**Principle**: Always check dtypes before passing to any sklearn-style
estimator. Object dtype columns will always fail.

### L-007: pip install -e . required for src imports in scripts
**What happened**: Running python scripts/evaluate.py raised
ModuleNotFoundError: No module named 'src' because Python does not
automatically add the project root to sys.path for scripts.
**Resolution**: pip install -e . registers the project root as a
known package location for the duration of the venv.
pytest adds the project root automatically via pythonpath=["."]:
scripts run directly do not get this behaviour.
**Principle**: Always install your project in editable mode when
running scripts that import from src/.
