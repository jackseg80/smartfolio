"""
ML Crypto Endpoints - Bitcoin Regime Detection API

Endpoints:
- GET /api/ml/crypto/regime: Current BTC regime (hybrid detection)
- GET /api/ml/crypto/regime-history: Historical regime timeline
"""

from fastapi import APIRouter, Query, HTTPException, Depends
from api.deps import get_required_user
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime
import logging
import time

from api.utils import success_response, error_response
from services.ml.models.btc_regime_detector import BTCRegimeDetector
from services.price_history import price_history
from services.regime_constants import REGIME_NAMES, smooth_regime_sequence

router = APIRouter(dependencies=[Depends(get_required_user)])
logger = logging.getLogger(__name__)

# Simple in-memory cache for regime history (TTL: 4 hours)
_regime_history_cache = {}
_CACHE_TTL = 14400  # 4 hours in seconds (historical data changes infrequently)


def get_btc_events(start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Dict[str, str]]:
    """
    Get significant Bitcoin events within date range for chart annotations.

    Args:
        start_date: Start of period
        end_date: End of period

    Returns:
        List of events with date, label, and type
    """
    all_events = [
        {'date': '2014-02-01', 'label': 'Mt.Gox Collapse', 'type': 'crisis'},
        {'date': '2014-12-14', 'label': 'Bear Bottom $315', 'type': 'bottom'},
        {'date': '2017-12-17', 'label': 'BTC ATH $20k', 'type': 'peak'},
        {'date': '2018-12-15', 'label': 'Crypto Winter Bottom $3.2k', 'type': 'bottom'},
        {'date': '2020-03-12', 'label': 'COVID Crash -50%', 'type': 'crisis'},
        {'date': '2020-12-16', 'label': 'BTC Breaks Previous ATH', 'type': 'policy'},
        {'date': '2021-04-14', 'label': 'Coinbase IPO', 'type': 'policy'},
        {'date': '2021-11-10', 'label': 'BTC ATH $69k', 'type': 'peak'},
        {'date': '2022-05-09', 'label': 'Luna/UST Collapse', 'type': 'crisis'},
        {'date': '2022-11-09', 'label': 'FTX Bankruptcy', 'type': 'crisis'},
        {'date': '2022-11-21', 'label': 'Bear Bottom $15.5k', 'type': 'bottom'},
        {'date': '2024-03-13', 'label': 'BTC New ATH $73k', 'type': 'peak'}
    ]

    # Filter events within date range
    filtered_events = []
    for event in all_events:
        event_date = pd.to_datetime(event['date'])
        if start_date <= event_date <= end_date:
            filtered_events.append(event)

    logger.info(f"Found {len(filtered_events)} BTC events in period {start_date.date()} to {end_date.date()}")
    return filtered_events


@router.get("/regime")
async def get_crypto_regime(
    symbol: str = Query("BTC", description="Cryptocurrency symbol"),
    lookback_days: int = Query(3650, ge=365, le=5000, description="Historical window for features (days)")
):
    """
    Get current Bitcoin market regime using hybrid detection.

    Uses rule-based + HMM fusion:
    - Rule-based: High confidence cases (bear >50% DD, bull stable)
    - HMM: Nuanced cases (corrections, consolidations)
    - Fusion: Rule overrides HMM if confidence ≥ 85%

    Returns:
        Current regime, confidence, detection method, probabilities
    """
    try:
        logger.info(f"GET /api/ml/crypto/regime - symbol={symbol}, lookback_days={lookback_days}")

        # Create detector
        detector = BTCRegimeDetector()

        # Predict current regime
        result = await detector.predict_regime(
            symbol=symbol,
            lookback_days=lookback_days,
            return_probabilities=True
        )

        # Format response
        response_data = {
            'current_regime': result['regime_name'],
            'confidence': result['confidence'],
            'detection_method': result['detection_method'],
            'rule_reason': result.get('rule_reason'),
            'regime_info': result['regime_info'],
            'regime_probabilities': result.get('regime_probabilities', {}),
            'benchmark': symbol,
            'lookback_days': lookback_days,
            'prediction_date': result['prediction_date'],
            'model_metadata': result['model_metadata'],
            'data_as_of': result.get('data_as_of')
        }

        response_data.update({key: result.get(key) for key in ("nature", "availability", "probability_kind", "hmm_state", "hmm_state_features", "economic_mapping_verified", "rule_diagnostic")})

        return success_response(response_data)

    except ValueError as e:
        logger.error(f"ValueError in get_crypto_regime: {e}")
        return error_response(str(e), code=400)
    except Exception as e:
        logger.error(f"Error in get_crypto_regime: {e}", exc_info=True)
        return error_response(f"Failed to predict regime: {str(e)}", code=500)


@router.get("/regime-history")
async def get_crypto_regime_history(
    symbol: str = Query("BTC", description="Cryptocurrency symbol"),
    lookback_days: int = Query(90, ge=30, le=3650, description="Timeline period (days)")
):
    """
    Get Bitcoin regime history using HYBRID detection (rule-based + HMM).

    Uses the same hybrid logic as current regime detection:
    - Rule-based for clear cases (Bear, Bull, Correction, Expansion)
    - HMM fallback for nuanced cases

    Args:
        symbol: Crypto symbol (default: BTC)
        lookback_days: Timeline length (30-3650 days, default 90)
    """
    try:
        logger.info(f"GET /api/ml/crypto/regime-history - symbol={symbol}, lookback_days={lookback_days}")

        # Check cache first
        cache_key = f"{symbol}_{lookback_days}"
        if cache_key in _regime_history_cache:
            cached_data, cache_time = _regime_history_cache[cache_key]
            if time.time() - cache_time < _CACHE_TTL:
                logger.debug(f"[Cache HIT] Returning cached regime history for {cache_key}")
                return success_response(cached_data)
            else:
                logger.debug(f"[Cache EXPIRED] Removing stale cache for {cache_key}")
                del _regime_history_cache[cache_key]

        # Get data
        history = price_history.get_cached_history(symbol, days=lookback_days)
        if history is None or len(history) == 0:
            return error_response(f"No historical data available for {symbol}", code=404)

        data = pd.DataFrame(history, columns=['timestamp', 'close'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index('timestamp', inplace=True)

        # Prepare features
        detector = BTCRegimeDetector()
        features_df = await detector.prepare_regime_features(symbol=symbol, lookback_days=lookback_days)

        if len(features_df) == 0:
            return error_response("Insufficient data", code=400)

        # Load symbol-specific HMM model for fallback
        model_file = f"{symbol.lower()}_regime_hmm.pkl"
        if not detector.load_model(model_file):
            return error_response("HMM history unavailable: explicit administrator training is required", code=503)

        features_scaled = detector.scaler.transform(features_df[detector.feature_columns])
        hmm_labels = detector.hmm_model.predict(features_scaled)

        # Pre-calculate rolling minimum drawdown for Expansion detection (performance optimization)
        features_df['lookback_180d_min_dd'] = features_df['drawdown_from_peak'].rolling(
            window=180, min_periods=1
        ).min()

        # Apply HYBRID detection for each day (rule-based + HMM fallback)
        regime_names = []
        regime_ids = []

        # Vectorized detection for simple rules (Bear, Bull, Correction)
        drawdown = features_df['drawdown_from_peak'].values
        days_since_peak = features_df['days_since_peak'].values
        trend_30d = features_df['trend_30d'].values
        volatility = features_df['market_volatility'].values
        lookback_dd = features_df['lookback_180d_min_dd'].values

        for i in range(len(features_df)):
            dd = drawdown[i]
            dsp = days_since_peak[i]
            trend = trend_30d[i]
            vol = volatility[i]
            lb_dd = lookback_dd[i]

            # Apply rule-based detection (optimized, no DataFrame slicing)
            rule_result = _detect_regime_rule_based_optimized(dd, dsp, trend, vol, lb_dd)

            if rule_result:
                regime_names.append(rule_result['regime_name'])
                regime_ids.append(rule_result['regime_id'])
            else:
                # Fallback to HMM
                hmm_label = int(hmm_labels[i])
                regime_names.append(f"State {chr(65+hmm_label)}")
                regime_ids.append(4 + hmm_label)

        # Smooth regime sequence to remove short-lived transitions (<7 days)
        regime_ids = smooth_regime_sequence(regime_ids, min_duration=7)
        label_mapping = {**dict(enumerate(REGIME_NAMES)), **{4+i: f"State {chr(65+i)}" for i in range(detector.num_regimes)}}
        regime_names = [label_mapping[rid] for rid in regime_ids]

        # Format response
        response_data = {
            'dates': features_df.index.strftime('%Y-%m-%d').tolist(),
            'prices': data.loc[features_df.index, 'close'].tolist(),
            'regimes': regime_names,
            'regime_ids': regime_ids,
            'symbol': symbol,
            'lookback_days': lookback_days,
            'regime_id_mapping': label_mapping,
            'economic_mapping_verified': False,
            'id_encoding': 'Rule diagnostics 0-3; unmapped HMM states 4-7',
            'events': get_btc_events(features_df.index.min(), features_df.index.max()),
            'note': 'Hybrid detection (rule-based + HMM fallback) with 7-day minimum duration smoothing.'
        }

        # Store in cache
        _regime_history_cache[cache_key] = (response_data, time.time())
        logger.debug(f"[Cache STORE] Cached regime history for {cache_key}")

        response_data["retrospective"] = True
        response_data["history_limitation"] = "Full-sequence HMM and smoothing use later observations; not real-time decision evidence"
        return success_response(response_data)

    except Exception as e:
        logger.error(f"Error in regime-history: {e}", exc_info=True)
        return error_response(f"Failed to get regime history: {str(e)}", code=500)


def _detect_regime_rule_based_for_row(row: pd.Series, features_context: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Apply rule-based regime detection for a single row with historical context.

    Crypto-adapted thresholds (aligned with _detect_regime_rule_based_optimized):
    - Bear: DD ≤ -30% + negative trend, sustained 20 days
    - Expansion: Recovery from -30%+ with +15%/month trend
    - Bull: DD > -20%, vol < 60%, trend > +10%
    - Bull (recovery): trend > +5%, vol < 65%, DD > -50%
    - Correction: DD < -10% + vol > 65%

    Returns:
        Dict with regime info if clear rule match, else None (defer to HMM)
    """
    drawdown = row.get('drawdown_from_peak', 0)
    days_since_peak = row.get('days_since_peak', 0)
    trend_30d = row.get('trend_30d', 0)
    volatility = row.get('market_volatility', 0)

    # Rule 1: BEAR MARKET - deep drawdown WITH clearly declining price
    if drawdown <= -0.30 and days_since_peak >= 20 and trend_30d <= -0.10:
        return {'regime_id': 0, 'regime_name': 'Bear Market'}

    # Rule 2: EXPANSION - recovering from deep drawdown (still far from ATH)
    # drawdown < -0.15 ensures we're still in recovery; near ATH → Bull Market instead
    lookback_dd = features_context.tail(180)['drawdown_from_peak'].min() if len(features_context) > 0 else 0
    if lookback_dd <= -0.30 and trend_30d >= 0.15 and drawdown < -0.15:
        return {'regime_id': 3, 'regime_name': 'Expansion'}

    # Rule 3: BULL MARKET - clear uptrend near peaks
    if drawdown >= -0.20 and volatility < 0.60 and trend_30d > 0.10:
        return {'regime_id': 2, 'regime_name': 'Bull Market'}

    # Rule 4: BULL MARKET (recovery) - moderate uptrend during recovery
    if trend_30d > 0.05 and volatility < 0.65 and drawdown > -0.50:
        return {'regime_id': 2, 'regime_name': 'Bull Market'}

    # Rule 5: CORRECTION - elevated vol OR deep drawdown with flat trend
    if (drawdown < -0.10 and volatility > 0.65) or (drawdown < -0.20 and abs(trend_30d) < 0.10):
        return {'regime_id': 1, 'regime_name': 'Correction'}

    # No clear rule-based detection → defer to HMM
    return None


def _detect_regime_rule_based_optimized(
    drawdown: float,
    days_since_peak: int,
    trend_30d: float,
    volatility: float,
    lookback_180d_min_dd: float
) -> Optional[Dict[str, Any]]:
    """
    Optimized version of rule-based detection using pre-calculated values.

    Args:
        drawdown: Current drawdown from peak
        days_since_peak: Days since last peak
        trend_30d: 30-day trend
        volatility: Market volatility
        lookback_180d_min_dd: Pre-calculated 180-day rolling minimum drawdown

    Returns:
        Dict with regime info if clear rule match, else None
    """
    # Rule 1: BEAR MARKET - deep drawdown WITH clearly declining price
    # trend <= -10% required: a -6%/month for crypto is noise, not a bear market
    if drawdown <= -0.30 and days_since_peak >= 20 and trend_30d <= -0.10:
        return {'regime_id': 0, 'regime_name': 'Bear Market'}

    # Rule 2: EXPANSION - recovering from deep drawdown (still far from ATH)
    # drawdown < -0.15 ensures we're still in recovery; near ATH → Bull Market instead
    if lookback_180d_min_dd <= -0.30 and trend_30d >= 0.15 and drawdown < -0.15:
        return {'regime_id': 3, 'regime_name': 'Expansion'}

    # Rule 3: BULL MARKET - clear uptrend near peaks
    if drawdown >= -0.20 and volatility < 0.60 and trend_30d > 0.10:
        return {'regime_id': 2, 'regime_name': 'Bull Market'}

    # Rule 4: BULL MARKET (recovery) - moderate uptrend even with deeper ATH drawdown
    if trend_30d > 0.05 and volatility < 0.65 and drawdown > -0.50:
        return {'regime_id': 2, 'regime_name': 'Bull Market'}

    # Rule 5: CORRECTION - elevated volatility OR deep drawdown with flat trend
    if (drawdown < -0.10 and volatility > 0.65) or (drawdown < -0.20 and abs(trend_30d) < 0.10):
        return {'regime_id': 1, 'regime_name': 'Correction'}

    # No clear rule-based detection → defer to HMM
    return None


@router.get("/regime-forecast")
async def get_crypto_regime_forecast(symbol: str = Query("BTC"), lookback_days: int = Query(90, ge=30, le=365)):
    return success_response({"available": False, "availability": "Experimental", "nature": "forecast",
        "forecast": None, "symbol": symbol,
        "reason": "Regime transition and conditional scenario probabilities have no independent forecasting validation. Use current rule diagnostics and HMM latent-state posteriors separately."})


@router.get("/regime/validate")
async def validate_regime_detector(
    symbol: str = Query("BTC", description="Cryptocurrency symbol")
):
    """Legacy retrospective validation is unavailable without dated inference."""
    return success_response({"symbol": symbol, "availability": "Unavailable", "validation_state": "not_evaluable",
        "bear_market_recall": None, "results": [], "status": "Unavailable",
        "reason": "The legacy test queried current observations for historical events. It cannot establish historical recall or real-time decision quality."})

