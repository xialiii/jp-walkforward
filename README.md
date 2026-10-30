# Walk Forward Backtesting of Return Prediction Models

Can simple price and volume features rank Japanese large cap stocks by their next 5 day return, out of sample? This project builds a leakage tested walk forward pipeline and reports the answer honestly.

**Main finding:** No model showed a statistically reliable out of sample ranking ability. Ridge, Lasso and LightGBM did not beat naive baselines, and every top 5 strategy underperformed equal weight and index buy and hold after costs.

## Data
Daily adjusted prices from Yahoo Finance (yfinance) for 30 Japanese large cap stocks plus the TOPIX ETF (1306.T) as a benchmark, January 2016 to September 2026. The stock list was chosen by hand from today's large companies across sectors.

Data cleaning (src/clean_data.py):
1. Rows on market holidays (zero volume and High equal to Low) are removed, 689 rows.
2. Two data source errors were found by scanning for extreme daily moves and removed with documented reasons: 1306.T on 2026-03-30 and 31 (price drops to about one tenth, then recovers), and 8766.T after 2026-09-28 (artificial jump around a stock split date, adjustment appears inconsistent). Other large moves match real market events and were kept.

## Method
Features (src/features.py), all using data up to day t only: momentum over 5, 20 and 60 days, 20 day volatility, 20 day volume change. Target: return over the next 5 trading days. For model training the target is demeaned across stocks each day, so models learn ranking rather than market direction.

Validation (src/walkforward.py): expanding window walk forward. Training starts in 2016, the first test year is 2021, and each test window is one year, 6 windows in total. Training data always precede the test data, with a 5 day gap because the target looks 5 days ahead.

Models (src/models.py): random baseline, 20 day momentum baseline, ridge, Lasso, LightGBM. Hyperparameters were fixed in advance and not tuned on test data.

Evaluation: daily rank information coefficient (Spearman correlation between predicted and realized return ranks across stocks), and a backtest (src/backtest.py): every 5 trading days buy the 5 highest ranked stocks, equal weight, with 10 bps one way cost, compared with equal weight holding of all 30 stocks and with the TOPIX ETF.

## Results

Rank IC, 2021 to September 2026 (1398 test days):

| Model | Mean IC | t stat | Share of positive days |
|---|---|---|---|
| Random | -0.0015 | -0.31 | 0.499 |
| Momentum 20 | -0.0056 | -0.77 | 0.494 |
| Ridge | -0.0087 | -1.14 | 0.489 |
| Lasso | -0.0080 | -1.04 | 0.491 |
| LightGBM | -0.0056 | -1.03 | 0.506 |

Backtest, net of costs for the model rows:

| Strategy | Annual return | Sharpe | Max drawdown |
|---|---|---|---|
| Random top 5 | 8.4% | 0.48 | -29.9% |
| Momentum top 5 | 13.6% | 0.64 | -32.4% |
| Ridge top 5 | 8.5% | 0.45 | -32.5% |
| Lasso top 5 | 4.3% | 0.30 | -30.3% |
| LightGBM top 5 | 7.4% | 0.44 | -29.8% |
| Equal weight 30 stocks | 22.3% | 1.21 | -17.6% |
| TOPIX ETF buy and hold | 18.7% | 1.09 | -17.3% |

![Equity curves](results/equity_curves.png)

Interpretation: all mean ICs are close to zero with t statistics inside plus or minus 2, so there is no evidence of predictive power from these features. The momentum baseline has the highest Sharpe among the top 5 strategies, but its IC is not positive and the gap to the other strategies is within noise for a single historical path, so I do not read it as evidence that momentum works.

## Regularization in geophysics and in finance

Ridge regression and Lasso belong to the same family of ideas as regularized inversion in geophysics: fit noisy data while a penalty keeps the solution stable. In my research I used Occam style inversion and ModEM, so this section describes those methods as given in the literature and compares them with the models in this project.

### 1. Occam inversion: the smoothest model at a target misfit

Occam inversion (Constable et al., 1987 for 1D; deGroot-Hedlin and Constable, 1990 for 2D magnetotellurics) looks for the smoothest model that fits the data to within an expected tolerance, instead of the best fitting model, which tends to be the roughest:

$$U(m) = \lVert W d - W f(m) \rVert_2^2 + \mu \lVert R m \rVert_2^2$$

Here $f(m)$ is the nonlinear forward response, $W$ weights the data by their errors, and $R$ is a finite difference operator, so the penalty acts on model roughness (a first derivative; second derivatives are also possible). At each iteration the trade off parameter $\mu$ is found by a line search. Until the target misfit (RMS close to 1) can be reached, $\mu$ is chosen to minimize the misfit. Once it can be reached, $\mu$ is chosen so that the misfit equals the target, which gives the smoothest model that fits the data acceptably.

### 2. ModEM: penalty functional with a prior and a model covariance

ModEM (Egbert and Kelbert, 2012; Kelbert et al., 2014) minimizes a penalty functional with nonlinear conjugate gradients:

$$\Phi(m, d) = (d - f(m))^\top C_d^{-1} (d - f(m)) + \nu \, (m - m_0)^\top C_m^{-1} (m - m_0)$$

$C_d$ is the covariance of the data errors, $m_0$ is a prior model (the starting model if no prior is given), $C_m$ is a model covariance that acts as a smoothing operator, and $\nu$ is the trade off (damping) parameter. Larger covariance values give smoother models, and the smoothing and $\nu$ are chosen by the user and tested against the resulting misfit and model.

### 3. Ridge regression and Lasso belong to the same family

Ridge regression with features $X$ and target $y$:

$$\min_w \; \lVert X w - y \rVert_2^2 + \lambda \lVert w \rVert_2^2 \quad\Rightarrow\quad \hat w = (X^\top X + \lambda I)^{-1} X^\top y$$

| | Occam | ModEM | Ridge regression |
|---|---|---|---|
| Forward model | nonlinear $f(m)$, linearized each iteration | nonlinear $f(m)$ | linear, $X w$ |
| Data weighting | $W$ | $C_d^{-1}$ | none (standardized features) |
| Penalty | roughness $\lVert R m \rVert^2$ | $(m - m_0)^\top C_m^{-1} (m - m_0)$ | $\lVert w \rVert^2$ |
| Reference model | none | prior $m_0$ | zero |
| Trade off parameter | $\mu$ by line search to reach the target misfit | $\nu$ set and tested by the user | $\lambda$ fixed in advance |

Ridge is the special case with a linear operator, an identity penalty matrix and a zero reference. In terms of the singular value decomposition, each component of the ridge solution is damped by the filter factor $\sigma_i^2 / (\sigma_i^2 + \lambda)$. Directions that the data constrain poorly (small $\sigma_i$) are suppressed most, which trades a little bias for much lower variance. The same mechanism stabilizes ill posed inversions.

Lasso replaces the squared penalty by an L1 penalty:

$$\min_w \; \lVert X w - y \rVert_2^2 + \lambda \lVert w \rVert_1$$

It sets some coefficients exactly to zero and has no closed form solution. Its counterpart in inversion is a sparsity promoting L1 constraint.

Bayesian view: if the noise is Gaussian and the prior is $w \sim N(0, \tau^2 I)$, the ridge solution is the maximum a posteriori estimate with $\lambda = \sigma^2 / \tau^2$. The ModEM functional has the same structure, with $C_m / \nu$ as the prior covariance around $m_0$. Lasso corresponds to a Laplace prior.

### 4. Where the analogy stops

| | Geophysical inversion | Return prediction |
|---|---|---|
| Operator | $f(m)$ is a known physical forward operator | $X$ is a set of statistical features with no known law |
| Goal | Recover a plausible model of the Earth | Predict new, unseen data |
| Choosing the trade off | Target misfit from known data errors (Occam), or tested smoothing and damping (ModEM) | Time ordered out of sample validation |
| Noise | Data errors are available and roughly stationary | Non stationary, heavy tailed, very low signal to noise ratio |

### 5. What this means for this project

Ridge uses alpha 1.0 and Lasso uses alpha 1e-4, with standardized features. Both values were fixed in advance and not tuned on test data. In Occam inversion the trade off follows from a known noise level through the target misfit. Returns offer no such target, so the analogous step would be selecting alpha by time ordered validation inside each training window, which is left as future work.

The regularized linear models did not beat the naive baselines. This is consistent with a weak signal: regularization can stabilize an estimate, but it cannot create predictive information that the features do not contain.

### References
- Constable, S. C., Parker, R. L., Constable, C. G. (1987). Occam's inversion: a practical algorithm for generating smooth models from electromagnetic sounding data. Geophysics, 52, 289 to 300.
- deGroot-Hedlin, C., Constable, S. (1990). Occam's inversion to generate smooth, two dimensional models from magnetotelluric data. Geophysics, 55, 1613 to 1624.
- Egbert, G. D., Kelbert, A. (2012). Computational recipes for electromagnetic inverse problems. Geophysical Journal International, 189, 251 to 267.
- Kelbert, A., Meqbel, N., Egbert, G. D., Tandon, K. (2014). ModEM: a modular system for inversion of electromagnetic geophysical data. Computers and Geosciences, 66, 40 to 53.

## Limitations
1. Survivorship bias: the stocks are today's large companies, which flatters buy and hold results.
2. The sample period 2021 to 2026 was a strong period for Japanese equities, and the backtest is a single historical path.
3. t statistics are optimistic because 5 day returns on adjacent days overlap.
4. The backtest trades at the signal day close, which is more optimistic than real execution. Equal weight and ETF rows include no costs.
5. Only 5 simple price and volume features and 30 stocks. No fundamentals, no tuning of hyperparameters.
6. Yahoo Finance data can contain errors. Two were found, others may remain.

## Reproduce (tested on macOS, Python 3.9)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/clean_data.py
python src/features.py
python -m src.run_models
python -m src.run_backtest
pytest -q
```

## Tests
Tests check that features do not change when future prices are altered (no leakage), that the target is the forward return, that walk forward windows keep training before testing, that cleaning removes only the intended rows, and that backtest costs and drawdown are computed correctly.
