# ML reliability — protocol frozen before evaluation

Decision support for manual use only. No allocation integration, recurring training,
directional research, production publication or order execution is authorized here.

## Data and targets

Daily closes with documented provider, dataset identity and adjustment policy.
Crypto: 7/30 calendar days, annualization 365. Stocks: first scheduled session
on/after the calendar target, annualization 252; adjusted close policy required.
Target variance uses returns strictly after decision close through target close.
Missing observations are never backfilled. Legacy 1h/4h/1d targets are unavailable.

## Frozen comparison

Persistence (past realized volatility), EWMA lambda 0.94, Ridge alpha 1;
corrected LSTM is an additional candidate only with compatible causal features.
At least 730 training days, 183 calibration days, 183 test days; advance 183 days;
reserve final 365 days for confirmation. Purge using target end dates.
Transforms fit only on training. Select on mean QLIKE, check MAE and folds.
Learned candidate must improve QLIKE >=5%, win >=2/3 folds, and maintain
that improvement on untouched confirmation. Otherwise retain the simple reference.
90% intervals require separate calibration and confirmation coverage 85–95%.
No verified confirmation means no published forecast or probability.

## Existing research decisions

Reuse crypto-forecast-lot2 causal research and dated data, preserving its identities.
Directional models and ML alert predictor remain rejected/disabled.
Transformer correlation forecasts remain experimental. Historical correlations,
economic rules and cycle diagnostics are descriptive, not forecasting evidence.
HMM probabilities refer to latent states, not future price direction; reconstructed
histories are retrospective unless independently proven causal.

## Delivery gates

Contracts: complete, partial, absent, stale, zero, incompatible artifact, load error,
API error and expired session. Tenant/source isolation, no training during reads,
future mutation, purge, train-only normalization and reproducibility.
Desktop/mobile checks and required project tests. Reports must separate observed
facts, validation conclusions and unavailable capabilities. Deployment requires
separate approval, authenticated checks and rollback preparation.

## Fixed implementation parameters and interpretation

Causal features: RV7, RV30, RV90, EWMA variance (lambda 0.94), absolute log return.
All realized-volatility calculations use population standard deviation (ddof=0).
LSTM: sequence 14, one layer with hidden width 8, Softplus positive output,
12 full-batch epochs, Adam learning rate 0.01, seed 1729, deterministic CPU.
Ridge: alpha 1 with training-only mean/std; negative predictions clipped at zero.
QLIKE uses variance ratio y²/p² - log(y²/p²) - 1, numerical floor 1e-10.
MAE is measured in annualized volatility fractions.

The first calibration boundary includes 10 calendar days of slack to satisfy
730 observed training days after target-end purge even on exchange holidays.
Every boundary is purged independently by target end, including confirmation.
Expanding training advances 183 days. There is no hyperparameter search.
Calibration residual absolute errors define one symmetric 90% interval width;
lower bounds are clipped at zero. Confirmation, rather than calibration alone,
decides whether the interval may be published.

Retrospective confirmation is an out-of-development comparison within this run,
not evidence of genuinely unseen future performance. Runs repeated to repair
calendar boundaries, refresh provider data and verify metadata are recorded in the
delivery report. No method or hyperparameter was chosen after those repeats to
optimize confirmation results. Targets overlap; no significance claim is made.
Stock inputs use the vendor's current split/dividend adjusted history, not a
point-in-time corporate-action archive. Exchange calendars determine target sessions.
