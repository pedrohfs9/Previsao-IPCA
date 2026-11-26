# IPCA Forecast — High-Dimensional Models for Brazilian Inflation

A Python / scikit-learn study based on:

> Garcia, M. G. P., Medeiros, M. C., & Vasconcelos, G. F. R. (2017).
> *Real-time inflation forecasting with high-dimensional models: The case of Brazil.*
> International Journal of Forecasting, 33(3), 679–693.
> [doi:10.1016/j.ijforecast.2017.02.002](http://dx.doi.org/10.1016/j.ijforecast.2017.02.002)

The paper forecasts monthly Brazilian CPI inflation (IPCA) 1 to 12 months ahead using
~90 candidate predictors (macroeconomic series and FOCUS survey expectations) and compares
shrinkage, factor, tree-based and subset-regression models against simple benchmarks and the
FOCUS market survey.

This project leaves most of the economics aside and treats the paper as an end-to-end
**machine learning project**: clean the original data, build horizon-shifted targets and lagged
features, fit models for each horizon, and compare them against the FOCUS survey.

**Main result:** a complete subset regression (CSR) with 1–4 month lags beats the FOCUS
survey median at every horizon from 1 to 12 months, with an RMSE 4–10% lower. A cross-validated
LASSO does not beat FOCUS at any horizon.

## Method

1. **Cleaning** (`notebooks/01_data_exploration.ipynb`)
   - Columns are renamed by position using `references/data_dictionary.csv`, because the
     original names were truncated to 30 characters and several are duplicated.
   - A `0.0001` placeholder in `electricity_consumption_total` (April 2011) is replaced by the
     March 2011 value.
   - January and February 2003 are dropped: in several differenced columns these rows still hold
     levels instead of changes.
   - Output: `data/processed/BRinf_clean.csv`, 154 months (March 2003 – December 2015) × 92 columns.
2. **First models, 1 month ahead** (`notebooks/02_target_variable.ipynb`): benchmarks (last value,
   FOCUS), LASSO, LassoCV and random forest.
3. **Multi-horizon forecasting** (`src/ipca_forecast/`): one direct model per horizon h = 1…12.
4. **CSR and lags** (`notebooks/03_csr.ipynb`): the CSR built step by step for h = 3, then the
   full comparison for every horizon.

### Target and train/test split

For horizon `h`, the target in month `t` is the monthly IPCA of month `t + h`
(`ipca_mom.shift(-h)`), and a separate model is fitted for each horizon (direct forecasting).

The split is a single chronological 70/30 split (`train_preparation` in `features.py`). The last
`h − 1` training rows are dropped, because their targets fall inside the test period. Without
this gap, the model would learn from IPCA values that are not yet known when the first test
forecast is made. For the same reason, the LassoCV uses `TimeSeriesSplit(gap=h − 1)`.

### Benchmark

The FOCUS survey median for horizon `h` is the column `focus_med_t{h+1}`, because in row `t`
`focus_med_t1` already forecasts the IPCA of month `t` itself. Results are reported as the
model RMSE divided by the FOCUS RMSE (**below 1 = the model beats FOCUS**).

### Models

- **LassoCV** (`make_models` in `models.py`): standardized features, penalty chosen by
  time-series cross-validation.
- **Complete subset regression** (`run_csr` in `models.py`):
  1. rank all candidate variables by absolute correlation with the target in the training set
     and keep the top 25;
  2. fit an OLS regression for every combination of 4 of them (12,650 regressions);
  3. forecast with the average of all 12,650 regressions.
- **Lags** (`add_lags` in `features.py`): the value of every variable 1 to 4 months earlier is
  added as a new column (92 → 460 columns). The first 4 months are lost.

## Results

All results use the cleaned data with 4 lags (`compare_with_focus(add_lags(df, 4), range(1, 13))`).
The test forecasts cover the IPCA from April–July 2012 (depending on `h`) to December 2015:
42 to 45 months. The models are trained on 85 to 104 months.

| h  | RMSE FOCUS | RMSE LassoCV | RMSE CSR | LassoCV / FOCUS | CSR / FOCUS |
|----|-----------:|-------------:|---------:|----------------:|------------:|
| 1  | 0.185 | 0.207 | **0.169** | 1.119 | **0.912** |
| 2  | 0.242 | 0.250 | **0.217** | 1.034 | **0.898** |
| 3  | 0.250 | 0.264 | **0.239** | 1.057 | **0.957** |
| 4  | 0.256 | 0.311 | **0.245** | 1.217 | **0.959** |
| 5  | 0.261 | 0.333 | **0.237** | 1.277 | **0.909** |
| 6  | 0.260 | 0.298 | **0.239** | 1.148 | **0.920** |
| 7  | 0.261 | 0.281 | **0.249** | 1.076 | **0.953** |
| 8  | 0.263 | 0.297 | **0.248** | 1.131 | **0.943** |
| 9  | 0.266 | 0.340 | **0.251** | 1.280 | **0.945** |
| 10 | 0.269 | 0.339 | **0.255** | 1.263 | **0.949** |
| 11 | 0.271 | 0.332 | **0.256** | 1.224 | **0.946** |
| 12 | 0.273 | 0.342 | **0.258** | 1.252 | **0.946** |

RMSE is in percentage points of monthly IPCA (the sample mean is 0.49% per month).

### Conclusions

- **The CSR beats FOCUS at every horizon.** Averaging thousands of small regressions uses
  information from many variables while keeping each model too simple to overfit the 85 to
  104 training months.
- **LassoCV loses to FOCUS at every horizon.** At several horizons the cross-validation picks a
  penalty so large that every coefficient is zero, so the model only predicts the training mean.
  With few observations and a gap of up to 11 months, the cross-validation folds are very small.
  This matches the paper, where LASSO also loses to FOCUS from h = 3 on.
- **Lags helped the CSR, but less than they appear to.** Without lags, the CSR ratio ranged from
  0.97 to 1.16. For h = 3, the RMSE fell from 0.271 to 0.239, but most of that drop (to 0.245)
  comes from losing the first four months of 2003. Those months are unusual (IPCA fell from
  1.23% to −0.15% after the 2002 crisis) and distort the regressions. The lags themselves account
  for the rest (0.245 → 0.239).
- **At 1 month ahead (notebook 02, no lags)**, FOCUS was the best single forecast
  (RMSE 0.185), ahead of LassoCV (0.190), random forest (0.219) and repeating the last value
  (0.243).

### Limitations

- **Single split instead of a rolling window.** The paper re-estimates every model each month
  on the most recent 9 years. Here each model is fitted once, and the 70/30 split point (and
  therefore the test period) shifts slightly with `h`.
- **Short test period.** 42 to 45 months is little evidence. A 4–10% gain is encouraging, but no
  statistical test (such as the paper's model confidence set) was run.
- **Different horizon definitions.** The paper's h = 1 is a forecast made 5 days before the IPCA
  release, so its horizons are about one month shorter than the ones used here. Compare results
  with the paper's Table 1 as ratios to FOCUS only, not one horizon against the same number.
- **Pre-selection by correlation.** The paper ranks variables by the t-statistic of a
  single-variable regression. For a regression with one variable this gives the same ranking as
  absolute correlation.

## Project structure

```
Previsao-IPCA/
├── data/
│   ├── raw/              # Original data from the paper (.rda files)
│   └── processed/        # BRinf_clean.csv, generated by notebook 01 (git-ignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Renaming and cleaning
│   ├── 02_target_variable.ipynb    # First models, 1 month ahead
│   └── 03_csr.ipynb                # CSR, lags and the all-horizon comparison
├── src/ipca_forecast/    # Reusable project code (installable package)
│   ├── features.py       # train_preparation, add_lags
│   ├── models.py         # make_models (LassoCV), run_csr
│   ├── evaluation.py     # compare_with_focus
│   ├── config.py         # Placeholder, not used yet
│   ├── data.py           # Placeholder, not used yet
│   └── selection.py      # Placeholder, not used yet
├── references/
│   └── data_dictionary.csv   # Column names, paper variable numbers, inferred transformations
├── scripts/  models/  reports/  tests/   # Empty, reserved for future work
├── pyproject.toml        # Project metadata and dependencies (managed by uv)
└── README.md
```

## Data

The data comes from the paper's companion R package,
[gabrielrvsc/hdeconometrics](https://github.com/gabrielrvsc/hdeconometrics), and is stored in
`data/raw/`:

| File          | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `BRinf.rda`   | Monthly IPCA and 91 candidate predictors, January 2003 – December 2015 |
| `voldata.rda` | Another dataset shipped with the package (not used in this project) |

The package has no official per-column documentation. `references/data_dictionary.csv` maps
each column to the paper's variable tables (Tables 12 and 13) and records the inferred
transformation (level, % change or first difference) and how confident each mapping is.

The `.rda` files are read in Python with [`pyreadr`](https://github.com/ofajardo/pyreadr),
so no R installation is required.

## Setup and reproduction

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
uv sync
```

1. Run `notebooks/01_data_exploration.ipynb` to create `data/processed/BRinf_clean.csv`.
2. Run `notebooks/02_target_variable.ipynb` and `notebooks/03_csr.ipynb`, or build the results
   table directly:

```python
import pandas as pd
from ipca_forecast.features import add_lags
from ipca_forecast.evaluation import compare_with_focus

df = pd.read_csv("data/processed/BRinf_clean.csv", index_col=0, parse_dates=True)
compare_with_focus(add_lags(df, 4), range(1, 13))
```

The full table runs about 150,000 CSR regressions plus 12 LassoCV fits and takes more than
10 minutes.

### Development tools

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pytest          # run tests (none written yet)
```

## Roadmap

- [x] Load `BRinf.rda`, rename and clean the dataset
- [x] Direct-forecast targets for each horizon, with a leakage-free train/test gap
- [x] Benchmarks: last value and FOCUS
- [x] LASSO (LassoCV)
- [x] Random forest (1 month ahead only)
- [x] Complete subset regression
- [x] Lagged features
- [x] RMSE / MAE tables by horizon
- [ ] Rolling-window backtest
- [ ] Penalty chosen by BIC and adaptive LASSO
- [ ] Factor model with targeted predictors (PCA)
- [ ] Forecast combination
- [ ] Statistical comparison of forecasts (model confidence set)
- [ ] Rebuild the dataset from public sources (BCB SGS, FOCUS API, IBGE) and track live
      forecasts for 2027
- [ ] Unit tests

## License

Code: MIT. The data belongs to its original sources (Bloomberg and Banco Central do Brasil)
as distributed in the `hdeconometrics` package.
