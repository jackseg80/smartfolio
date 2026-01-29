# Exception Handling Guide - SmartFolio

> Guide des bonnes pratiques pour la gestion d'exceptions
> **Date**: 2026-01-29
> **Status**: Référence officielle

## Table des Matières

- [Hiérarchie d'Exceptions](#hiérarchie-dexceptions)
- [Patterns Recommandés](#patterns-recommandés)
- [Patterns à Éviter](#patterns-à-éviter)
- [Règles de Refactoring](#règles-de-refactoring)
- [Exemples Pratiques](#exemples-pratiques)

---

## Hiérarchie d'Exceptions

SmartFolio utilise deux hiérarchies d'exceptions personnalisées (préférez `shared/exceptions.py`) :

### Exceptions Disponibles

```python
from shared.exceptions import (
    # Base
    CryptoRebalancerException,     # Exception de base

    # Configuration
    ConfigurationError,             # Erreurs de configuration

    # API/Network
    APIException,                   # Erreurs d'API externe
    RateLimitException,             # Rate limiting
    NetworkException,               # Erreurs réseau
    TimeoutException,               # Timeouts

    # Data
    DataException,                  # Erreurs de données générales
    DataNotFoundException,          # Données non trouvées

    # Trading/Exchange
    ExchangeException,              # Erreurs d'exchange
    InsufficientBalanceException,   # Solde insuffisant

    # Pricing
    PricingException,               # Erreurs de pricing
)
```

### ErrorCode Enum

```python
from shared.exceptions import ErrorCode

# Codes disponibles:
ErrorCode.CONFIG_INVALID
ErrorCode.API_UNAVAILABLE
ErrorCode.DATA_NOT_FOUND
ErrorCode.NETWORK_ERROR
ErrorCode.TIMEOUT_ERROR
# ... voir shared/exceptions.py pour la liste complète
```

---

## Patterns Recommandés

### ✅ Pattern 1: Catches Spécifiques en Cascade

**Recommandé** : Catcher les exceptions connues d'abord, `Exception` en dernier recours uniquement.

```python
from shared.exceptions import DataException, NetworkException, convert_standard_exception

def fetch_portfolio_data(user_id: str):
    try:
        data = external_api.fetch(user_id)
        return process_data(data)

    except DataNotFoundException as e:
        logger.warning(f"Portfolio not found for user {user_id}: {e}")
        return {}  # Safe fallback

    except NetworkException as e:
        logger.error(f"Network error fetching portfolio: {e}")
        raise  # Re-raise si on ne peut pas gérer

    except Exception as e:
        # Catch-all de dernier recours uniquement
        logger.exception(f"Unexpected error in fetch_portfolio_data: {e}")
        converted = convert_standard_exception(e, "fetch_portfolio_data")
        raise converted
```

**✅ Pourquoi c'est acceptable** :
- Exceptions spécifiques catchées en premier
- `Exception` comme safety net de dernier recours
- Logging avec stacktrace complet (`logger.exception`)
- Conversion en exception typée avec `convert_standard_exception()`

---

### ✅ Pattern 2: Fallback Sécurisé avec Logging

**Recommandé** : Si vous devez retourner un fallback sûr en cas d'erreur inattendue.

```python
from shared.exceptions import GovernanceException

def derive_execution_policy(signals: dict) -> Policy:
    try:
        # Logique complexe de dérivation de policy
        phase = calculate_phase(signals)
        risk_score = calculate_risk(signals)
        return build_policy(phase, risk_score)

    except (ValueError, KeyError) as e:
        # Données invalides - fallback contrôlé
        logger.warning(f"Invalid signals data, using safe policy: {e}")
        return Policy(mode="Conservative", cap_daily=0.05)

    except Exception as e:
        # Erreur critique inattendue - fallback ultra-safe
        logger.exception(f"Critical error deriving policy, freezing: {e}")
        return Policy(mode="Freeze", cap_daily=0.08, notes=f"Error fallback: {e}")
```

**✅ Pourquoi c'est acceptable** :
- Système critique nécessitant un fallback sûr en toutes circonstances
- Logging avec stacktrace (`logger.exception`)
- Fallback explicitement marqué dans les notes
- Mode "Freeze" = position défensive claire

---

### ✅ Pattern 3: Helper de Conversion

**Recommandé** : Utiliser `convert_standard_exception()` pour auto-conversion.

```python
from shared.exceptions import convert_standard_exception

async def fetch_price(symbol: str) -> float:
    try:
        return await external_api.get_price(symbol)
    except Exception as e:
        # Conversion automatique vers exception typée
        converted = convert_standard_exception(e, f"fetch_price({symbol})")
        raise converted
```

**Helper disponible** :
```python
convert_standard_exception(exc, context="operation_name")
```

Convertit automatiquement :
- `ConnectionError`, `OSError` → `NetworkException`
- `TimeoutError` → `TimeoutException`
- `ValueError`, `KeyError`, `TypeError` → `DataException`
- `PermissionError` → `CryptoRebalancerException` avec `ErrorCode.PERMISSION_DENIED`

---

## Patterns à Éviter

### ❌ Anti-Pattern 1: Bare `except Exception` Sans Catches Spécifiques

**Problème** : Masque toutes les erreurs sans distinction.

```python
# ❌ À ÉVITER
def process_data(data):
    try:
        return complex_calculation(data)
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
```

**✅ Correction** :

```python
# ✅ CORRIGÉ
def process_data(data):
    try:
        return complex_calculation(data)
    except (ValueError, TypeError) as e:
        # Erreurs de données attendues
        logger.warning(f"Invalid data format: {e}")
        raise DataException(f"Cannot process data: {e}", data_source="user_input")
    except ZeroDivisionError as e:
        logger.error(f"Calculation error (division by zero): {e}")
        raise DataException(f"Calculation error: {e}", data_source="calc_engine")
    except Exception as e:
        # Seulement en dernier recours
        logger.exception(f"Unexpected error processing data: {e}")
        raise convert_standard_exception(e, "process_data")
```

---

### ❌ Anti-Pattern 2: Silent Failure

**Problème** : Catch sans logging ni action.

```python
# ❌ À ÉVITER
def load_config():
    try:
        with open("config.json") as f:
            return json.load(f)
    except Exception:
        return {}  # Silent failure - aucun log !
```

**✅ Correction** :

```python
# ✅ CORRIGÉ
def load_config():
    try:
        with open("config.json") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info("Config file not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Config file is invalid JSON: {e}")
        raise ConfigurationError("Invalid config.json format", cause=e)
    except PermissionError as e:
        logger.error(f"Cannot read config.json: permission denied")
        raise ConfigurationError("Cannot access config.json", cause=e)
```

---

### ❌ Anti-Pattern 3: Masquer les Stack Traces

**Problème** : `logger.error()` au lieu de `logger.exception()` pour erreurs inattendues.

```python
# ❌ À ÉVITER
def critical_operation():
    try:
        do_something_complex()
    except Exception as e:
        logger.error(f"Error: {e}")  # Perd le stacktrace !
        raise
```

**✅ Correction** :

```python
# ✅ CORRIGÉ
def critical_operation():
    try:
        do_something_complex()
    except DataException as e:
        # Exception connue - logger.error suffisant
        logger.error(f"Data error in critical_operation: {e}")
        raise
    except Exception as e:
        # Exception inattendue - stacktrace complet nécessaire
        logger.exception(f"Unexpected error in critical_operation: {e}")
        raise
```

**Règle** : Utilisez `logger.exception()` dans les blocs `except Exception` pour capturer le stacktrace complet.

---

## Règles de Refactoring

### Quand Garder `except Exception`

✅ **Acceptable** dans ces cas :

1. **Après des catches spécifiques** (catch-all de dernier recours)
2. **Systèmes critiques** nécessitant un fallback sûr (Governance, AlertEngine)
3. **Avec logging complet** (`logger.exception()`)
4. **Avec conversion** vers exception typée (`convert_standard_exception()`)
5. **Fallback explicite** documenté dans le code

### Quand Remplacer `except Exception`

🔄 **À refactorer** si :

1. **Seul catch** dans le bloc try/except
2. **Pas de logging** ou logging incomplet
3. **Erreurs prévisibles** qui devraient être catchées spécifiquement
4. **Masque des bugs** réels au lieu de les laisser remonter

### Checklist de Refactoring

Avant de refactorer un `except Exception`, vérifiez :

- [ ] Quelles exceptions spécifiques peuvent être levées dans le `try` ?
- [ ] Est-ce que je peux catcher ces exceptions spécifiquement ?
- [ ] Est-ce que je log avec `logger.exception()` ?
- [ ] Est-ce qu'il y a un fallback sécurisé documenté ?
- [ ] Est-ce que je convertis en exception typée ?

---

## Exemples Pratiques

### Exemple 1: API Externe

```python
from shared.exceptions import APIException, NetworkException, convert_standard_exception
import httpx

async def fetch_exchange_rate(base: str, quote: str) -> float:
    """Fetch exchange rate from external API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.example.com/rate/{base}/{quote}",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return float(data["rate"])

    except httpx.TimeoutException:
        raise TimeoutException("fetch_exchange_rate", timeout_seconds=10)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise RateLimitException("exchange_api", retry_after=60)
        elif e.response.status_code == 404:
            raise DataNotFoundException("exchange_rate", identifier=f"{base}/{quote}")
        else:
            raise APIException(f"HTTP {e.response.status_code}", api_name="exchange_api")

    except httpx.NetworkError as e:
        raise NetworkException(f"Network error fetching rate: {e}", url=str(e.request.url))

    except (ValueError, KeyError) as e:
        raise DataException(f"Invalid response format: {e}", data_source="exchange_api")

    except Exception as e:
        # Catch-all de dernier recours
        logger.exception(f"Unexpected error fetching exchange rate {base}/{quote}: {e}")
        raise convert_standard_exception(e, f"fetch_exchange_rate({base}/{quote})")
```

**Points forts** :
- ✅ Catches spécifiques pour tous les cas prévisibles
- ✅ Exceptions personnalisées avec contexte
- ✅ `Exception` en dernier recours seulement
- ✅ Conversion automatique avec `convert_standard_exception()`

---

### Exemple 2: Gouvernance avec Fallback

```python
from shared.exceptions import GovernanceException

def calculate_daily_cap(signals: dict, risk_level: str) -> float:
    """Calculate safe daily trading cap based on signals"""
    try:
        # Logique complexe de calcul
        market_phase = signals["phase"]["current"]
        risk_score = signals["risk"]["composite"]
        volatility = signals["volatility"]["current"]

        if risk_level == "conservative":
            base_cap = 0.03
        elif risk_level == "moderate":
            base_cap = 0.05
        elif risk_level == "aggressive":
            base_cap = 0.10
        else:
            raise ValueError(f"Invalid risk_level: {risk_level}")

        # Ajustements basés sur market conditions
        phase_multiplier = get_phase_multiplier(market_phase)
        risk_multiplier = get_risk_multiplier(risk_score)
        vol_multiplier = get_volatility_multiplier(volatility)

        final_cap = base_cap * phase_multiplier * risk_multiplier * vol_multiplier

        # Clamp entre limites sécuritaires
        return max(0.01, min(final_cap, 0.15))

    except (KeyError, ValueError, TypeError) as e:
        # Données invalides - fallback conservateur
        logger.warning(f"Invalid signals for daily cap calculation, using conservative 3%: {e}")
        return 0.03

    except Exception as e:
        # Erreur inattendue - fallback ultra-conservateur
        logger.exception(f"Unexpected error calculating daily cap, using minimal 1%: {e}")
        return 0.01
```

**Points forts** :
- ✅ Catches spécifiques pour erreurs de données attendues
- ✅ Fallback explicite et sécurisé (3% conservative, 1% emergency)
- ✅ Logging avec stacktrace (`logger.exception`)
- ✅ Valeurs de retour sécuritaires documentées

---

### Exemple 3: Storage avec Retry

```python
from shared.exceptions import StorageException
import redis

def save_to_redis(key: str, value: dict, ttl_seconds: int = 300) -> bool:
    """Save data to Redis with automatic retry"""
    try:
        redis_client = get_redis_client()
        serialized = json.dumps(value)
        redis_client.setex(key, ttl_seconds, serialized)
        return True

    except redis.ConnectionError as e:
        raise StorageException(
            storage_type="redis",
            operation="save",
            message=f"Cannot connect to Redis: {e}"
        )

    except redis.TimeoutError as e:
        raise TimeoutException("redis_save", timeout_seconds=5)

    except (json.JSONEncodeError, TypeError) as e:
        raise DataException(f"Cannot serialize value to JSON: {e}", data_source="redis_save")

    except Exception as e:
        logger.exception(f"Unexpected error saving to Redis key={key}: {e}")
        raise StorageException(
            storage_type="redis",
            operation="save",
            message=f"Unexpected error: {e}"
        )
```

---

## Statistiques Projet

**État actuel** (2026-01-29) :

- **729** occurrences de `except Exception` dans le projet
- **Top fichiers** :
  - `services/execution/governance.py` (37)
  - `services/alerts/alert_storage.py` (37)
  - `services/execution/exchange_adapter.py` (24)
  - `services/alerts/alert_engine.py` (24)
  - `services/monitoring/phase3_health_monitor.py` (23)

**Stratégie** : Refactoring progressif, pas de Big Bang. Prioriser les nouveaux codes et les fichiers critiques.

---

## Tooling

### Linter Configuration (TODO)

Configuration flake8/ruff pour détecter les patterns problématiques :

```toml
# TODO: À ajouter dans pyproject.toml
[tool.ruff]
select = ["BLE001"]  # Do not catch blind exception: `Exception`

# Exceptions autorisées dans certains contextes
[tool.ruff.per-file-ignores]
"services/execution/governance.py" = ["BLE001"]  # Fallback sécurisé critique
"services/alerts/alert_engine.py" = ["BLE001"]   # Fallback sécurisé critique
```

---

## Conclusion

**Règle d'or** : Évitez `except Exception` sauf en dernier recours avec logging complet.

**Ordre de préférence** :
1. ✅ Catches spécifiques (`ValueError`, `KeyError`, etc.)
2. ✅ Exceptions personnalisées (`DataException`, `APIException`, etc.)
3. ✅ `convert_standard_exception()` pour auto-conversion
4. ⚠️ `except Exception` uniquement comme safety net avec `logger.exception()`

**Questions ?** Consultez `shared/exceptions.py` ou demandez une review.
