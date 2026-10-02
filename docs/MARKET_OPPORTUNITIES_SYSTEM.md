# Market Opportunities v2

Manual stock-portfolio decision support. The authenticated scan explains source evidence, industry uncertainty and historical candidate screens. It does not generate broker orders or validated forward-return forecasts.

## Contract and source selection

- `GET /api/bourse/opportunities`: authenticated `get_required_user`, no-store JSON envelope, `data.version=2`.
- Parameters: `source`, `file_key`, `horizon`, `candidate_sector`, `sector_targets`, `min_gap_pct`.
- Active source and selected Saxo CSV come from the authenticated user's existing configuration. A conflicting request is rejected. A missing file never selects a newer CSV or demo account.
- CSV positions retain ISIN, exact listing, quote currency, asset class and quantity. Value currency comes from the original header, before the legacy connector removes its unit. Current verified FX converts those dated source values to USD. No indicative FX or parity fallback is accepted.
- Missing source valuations remain `null`, with the position retained. They are never turned into zero. Unknown values block full-portfolio sector bounds, concentration percentages and scenario valuation.
- Cash comes only from `<selected_csv>_cash.json`. Missing cash remains unavailable. USD cash and non-USD cash are dated separately from securities. Current verified FX is used for non-USD cash.
- Import/file dates are labeled `imported_at`; financial `valuation_as_of` remains null when the CSV does not provide a trustworthy valuation date. Acquisition dates are not inferred from notification/value/import dates.
- Manual positions are supported with an explicit USD value contract; cash remains unavailable. Saxo API positions require an explicit USD valuation contract; unavailable contracts fail visibly rather than guess units.
- Private snapshots are request-scoped and never stored in a shared classification/result cache. `snapshot_id` prevents using a scenario after changes to the user, CSV contents/selection, cash record or verified FX.

## Industry evidence and independent dimensions

`fund_exposure.py` accepts a fixed public iShares product catalog by exact ISIN. The adapter checks the matching page identity, the **Fund** table (never Benchmark), its own observation date, finite non-negative weights, duplicate sectors, and total weights. The observation must be no more than 45 days old and not future-dated. Receipts include issuer URL, observation/retrieval dates and a public-document hash.

Supported public identities:

- iShares Core MSCI World UCITS ETF: IE00B4L5Y983, [issuer product page](https://www.ishares.com/uk/individual/en/products/251882/ishares-msci-world-ucits-etf-acc-fund).
- iShares Core MSCI EM IMI UCITS ETF: IE00BKM4GZ66, [issuer product page](https://www.ishares.com/uk/individual/en/products/264659/ishares-core-msci-em-imi-ucits-etf).

Only visible sector rows are accepted. Omitted rows, cash and unrecognized labels stay unclassified; partial totals are not scaled to 100%. Unsupported funds expose the missing adapter explicitly. An index factsheet is not treated as actual fund composition. Issuer failures or changed page shapes do not invent weights.

Individual equities use a secondary Yahoo company-sector profile only when its listing and quote currency match. The URL and retrieval date are shown. An effective sector date is unavailable, and this weaker evidence is labeled separately from dated issuer look-through.

Geography is independent of industries. Company domicile is labeled as domicile, not revenue exposure. No verified fund geography adapter is presently included. Asset classes remain source metadata; broad ETF names and geographic labels never become economic sectors. Non-industry fund exposures remain outside industry coverage. An equity-only denominator is not guessed.

Coverage is explicitly scoped: all valued securities, or only the known-value subset when some positions lack values. The subset coverage is not full-portfolio coverage.

## Targets and conservative intervals

Default industry targets reproduce the legacy normalized midpoint of static sector ranges. They are a **generic reference**, with no current-index or personal-policy claim. Complete user-entered targets total 100% and are labeled `personal_policy`.

With complete valuations, an industry's lower bound is its verified allocation. Its upper bound adds all unclassified securities weight, capped at 100%. A sector is certainly underweight only when `target - upper_bound` exceeds the selected threshold; certainly overweight only when `lower_bound - target` exceeds it. Unknown composition can make all assessments Indeterminate. Missing position valuations make all full-portfolio bounds unavailable. Zero certain gaps does not establish balance.

Targets use all valued securities excluding saved cash, not an inferred equity sleeve. This scope is shown beside the target controls and in responses.

## Candidate exploration and historical ranking

Exploration runs independently of certain gaps. The default universe contains the eleven curated sector ETFs. Choosing one sector additionally examines all stocks in that sector's existing curated list. This is not the whole investable market or a suitability screen.

Exact mapped held listings are excluded. A matching candidate ISIN, when the provider supplies it, also excludes a different listing of the same instrument. Missing candidate ISIN leaves that deduplication incomplete. Fund constituent overlap is unassessed and disclosed. Stock candidates require their actual reported industry to match the curated industry. ETF names describe mandate; they do not establish a 100% industry composition.

Prices come from the existing verified adjusted-price pipeline, with no synthetic or proxy series. Today is excluded, listing quote currencies are checked, and complete sessions through the latest completed exchange date are required. Unsupported calendars, holes, stale observations, ambiguous currencies and insufficient history produce explicit reasons. Non-USD histories use dated actual FX with the existing maximum three-day backward alignment tolerance, not current spot FX.

Historical windows: short 42 sessions, medium 189, long 756. Their associated holding horizons remain planning labels, not tested return predictions.

```
score = clip(50 + 15 * mean(daily USD returns)
             / sample_std(daily USD returns) * sqrt(window_sessions), 0, 100)
```

A full window plus one starting close is required. Return, annualized historical volatility, dates, currency and method accompany the score. No value/dividend-yield composite, confidence probability or invented diversification component is added. Ranks are suppressed if candidates end on different dates. Scores describe historical adjusted prices and are not forward validated.

## Holding reviews and manual scenarios

Holding reviews show generic concentration triggers (10% only with complete valuations), missing classification/valuation evidence and missing acquisition data. They do not turn unknown holding periods or unknown stop orders into automatic sale permission. Automatic sales remain empty with a specific eligibility explanation; absence of a sale is not a favorable portfolio judgment.

`POST /api/bourse/opportunities/scenario` performs read-only arithmetic. No persistence, order service, broker execution, commit or deployment is called. The request contains the selected source/settings and a manual `scenario` with:

- current `snapshot_id`;
- 1–50 unique changes: `side`, stable holding/candidate `id`, positive finite `amount_usd`;
- explicit non-negative `costs_usd`, `slippage_pct` from 0 to 10;
- `acknowledge_dated_values: true`;
- optional boolean `include_history`.

Sales cannot exceed held snapshot value. Purchases must be current screened candidates. Combined purchases and costs cannot exceed saved cash plus sales. Securities plus cash after the changes must equal before total minus estimated friction. No executable share quantity, purchase capital recommendation, fund-composition assumption, tax-lot rule or stop-order assurance is created. Purchased ETFs remain unclassified without a verified decomposition.

Optional historical risk requires exact USD histories for **all** nonzero positions in both scenarios and common return intervals with the same start and end dates. The full selected signal window and at least 60 daily intervals are required. Historical volatility uses fixed current USD weights, zero-return cash and a daily-rebalanced approximation. Candidate correlation is measured against the actual before portfolio on those same intervals. Partial holdings are not used as a portfolio proxy. Missing histories leave volatility unavailable with counts/reasons. No validated 0–100 Risk Score is implemented.

## Frontend and confidentiality

The Market Opportunities tab uses `static/components/market-opportunities.js`. It checks authenticated account/source/CSV context and request revision before and after asynchronous body decoding, clears obsolete results and prevents markup injection through provider text or URLs. All visible labels and errors are in English.

The existing Recommendations tab and its independent ML work are preserved. API v2 intentionally replaces the old opportunities payload; external clients must migrate to the versioned contract. Export returns only authorized aggregate audit controls and method metadata, never holdings, amounts, account identifiers, file names, private hashes or keys.

## Validation and open data work

Synthetic test fixtures verify strict CSV selection, currencies, null valuations, partial issuer receipts, interval bounds, identity matching, real-history requirements and cash conservation. API tests verify authenticated context, validation and no-store responses. DOM tests cover missing evidence, provider text and late responses. Public issuer pages and US/Swiss price series are checked separately from a read-only real-account audit.

Remaining data work: issuer adapters for unsupported funds, trustworthy financial valuation dates/complete CSV values, exact candidate ISIN coverage, fund constituent overlap, personal tax/holding/stop-order constraints, personal target policy, and fund geography. Their absence stays visible. No new provider account, credentials, environment or private data image is needed for the implemented adapters.
