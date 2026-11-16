# IPCA Forecast — High-Dimensional Models for Brazilian Inflation

A Python / scikit-learn reproduction of:

> Garcia, M. G. P., Medeiros, M. C., & Vasconcelos, G. F. R. (2017).
> *Real-time inflation forecasting with high-dimensional models: The case of Brazil.*
> International Journal of Forecasting, 33(4), 679–693.
> [doi:10.1016/j.ijforecast.2017.02.002](http://dx.doi.org/10.1016/j.ijforecast.2017.02.002)

The paper forecasts monthly Brazilian CPI inflation (IPCA) 1 to 12 months ahead using
~90 candidate predictors (macroeconomic series and FOCUS survey expectations) and compares
shrinkage, factor, tree-based and subset-regression models against simple benchmarks.

This project leaves most of the economics aside and treats the paper as an end-to-end
**machine learning project**:

1. **Data loading:** read the original `.rda` files into pandas
2. **Preprocessing:** clean the data and build lagged features and horizon-shifted targets
3. **Feature selection:** pre-test candidate predictors (targeted predictors)
4. **Modeling:** fit scikit-learn models in a rolling-window backtest
5. **Evaluation:** compare RMSE / MAE across models and forecast horizons

## Project structure

```
Previsao-IPCA/
├── data/
│   ├── raw/              # Original, immutable data (.rda files from the paper)
│   └── processed/        # Cleaned / feature-engineered datasets (generated, git-ignored)
├── notebooks/            # Exploration and step-by-step analysis
├── src/ipca_forecast/    # Reusable project code (installable package)
│   ├── config.py         # Paths and experiment settings
│   ├── data.py           # Data loading
│   ├── features.py       # Preprocessing and feature engineering
│   ├── selection.py      # Feature selection
│   ├── models.py         # Benchmarks and scikit-learn models
│   └── evaluation.py     # Backtest, error metrics, forecast combination
├── scripts/              # Command-line entry points (run the full pipeline)
├── models/               # Serialized fitted models (generated, git-ignored)
├── reports/
│   ├── figures/          # Generated plots (git-ignored)
│   └── tables/           # Generated result tables (git-ignored)
├── references/           # Paper and supporting material
├── tests/                # Unit tests (pytest)
├── pyproject.toml        # Project metadata and dependencies (managed by uv)
└── README.md
```

## Data

The data comes from the paper's companion R package,
[gabrielrvsc/hdeconometrics](https://github.com/gabrielrvsc/hdeconometrics), and is stored in
`data/raw/`:

| File          | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `BRinf.rda`   | Monthly IPCA and candidate predictors used in the paper            |
| `voldata.rda` | Additional dataset shipped with the package                        |

The `.rda` files are read in Python with [`pyreadr`](https://github.com/ofajardo/pyreadr),
so no R installation is required.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
# Create the virtual environment and install all dependencies (including dev tools)
uv sync

# Run any command inside the environment
uv run python -c "import ipca_forecast"

# Add a new dependency
uv add <package>
```

A `requirements.txt` for pip users can be exported from the lockfile:

```bash
uv export --no-hashes --no-dev -o requirements.txt
```

### Development tools

```bash
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Roadmap

- [ ] Load `BRinf.rda` and explore the dataset
- [ ] Build lagged features and direct-forecast targets for each horizon
- [ ] Implement the rolling-window backtest
- [ ] Benchmarks: random walk and AR
- [ ] LASSO and adaptive LASSO
- [ ] Random forest
- [ ] Factor model with targeted predictors (PCA)
- [ ] Complete subset regression
- [ ] Error tables (RMSE / MAE) by horizon and comparison with the paper's Table 1
- [ ] Forecast combination

## Results

_To be added._

## License

Code: MIT. The data belongs to its original sources (Bloomberg and Banco Central do Brasil)
as distributed in the `hdeconometrics` package.
