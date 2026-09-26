# Guide de Migration: Exception Handling

**Date:** 29 Octobre 2025
**Objectif:** Remplacer les `except Exception` trop larges par des exceptions spécifiques
**Impact:** Amélioration du debugging, meilleure gestion d'erreurs, moins de bugs masqués

---

## État Actuel

**Problème:** 109 occurrences de `except Exception` dans le projet masquent des bugs et rendent le debugging difficile.

**Top 5 fichiers critiques:**
1. `services/execution/governance.py` - **42 occurrences**
2. `services/alerts/alert_storage.py` - **37 occurrences**
3. `services/execution/exchange_adapter.py` - **24 occurrences**
4. `services/alerts/alert_engine.py` - **24 occurrences**
5. `services/monitoring/phase3_health_monitor.py` - **23 occurrences**

---

## Exceptions Disponibles

Fichier: `api/exceptions.py`

### Hiérarchie

```
CryptoRebalancerException (base)
├── APIException              # Erreurs d'API externe
├── ValidationException       # Erreurs de validation
├── ConfigurationException    # Erreurs de configuration
├── TradingException          # Erreurs de trading/rebalancing
├── DataException             # Erreurs de données
├── StorageException          # Erreurs de stockage (Redis, fichiers)
├── GovernanceException       # Erreurs de gouvernance
├── MonitoringException       # Erreurs de monitoring
└── ExchangeException         # Erreurs d'exchange adapters
```

### Usage

```python
from api.exceptions import (
    APIException, ValidationException, ConfigurationException,
    StorageException, GovernanceException, DataException
)
```

---

## Patterns de Migration

### Pattern 1: Initialisation de Composants

**[Error] AVANT:**
```python
try:
    from ..ml.orchestrator import get_orchestrator
    ML_ORCHESTRATOR_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    ML_ORCHESTRATOR_AVAILABLE = False
```

**[OK] APRÈS:**
```python
from api.exceptions import ConfigurationException

try:
    from ..ml.orchestrator import get_orchestrator
    ML_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML Orchestrator not available: {e}")
    ML_ORCHESTRATOR_AVAILABLE = False
except (AttributeError, ModuleNotFoundError) as e:
    raise ConfigurationException(f"Failed to initialize ML components: {e}")
```

**Règle:** Imports manquants = `ImportError`, problèmes de config = `ConfigurationException`

---

### Pattern 2: Appels API Externes

**[Error] AVANT:**
```python
try:
    response = await httpx.get(url)
    data = response.json()
except Exception as e:
    logger.error(f"Failed to fetch data: {e}")
    return {}
```

**[OK] APRÈS:**
```python
from api.exceptions import APIException

try:
    response = await httpx.get(url, timeout=10.0)
    response.raise_for_status()
    data = response.json()
except httpx.HTTPError as e:
    raise APIException(
        service="external_api",
        message=f"HTTP request failed: {e}",
        status_code=getattr(e.response, 'status_code', None)
    )
except httpx.TimeoutException as e:
    raise APIException(service="external_api", message="Request timeout", status_code=503)
except ValueError as e:  # JSON decode error
    raise DataException(source="external_api", message=f"Invalid JSON response: {e}")
```

**Règle:** Erreurs HTTP = `APIException`, problèmes de données = `DataException`

---

### Pattern 3: Validation de Données

**[Error] AVANT:**
```python
try:
    value = float(user_input)
    if value < 0:
        raise ValueError("Negative value")
except Exception as e:
    logger.error(f"Validation failed: {e}")
    return default_value
```

**[OK] APRÈS:**
```python
from api.exceptions import ValidationException

try:
    value = float(user_input)
except ValueError as e:
    raise ValidationException(
        field="user_input",
        message=f"Must be a valid number: {e}",
        value=user_input
    )

if value < 0:
    raise ValidationException(
        field="user_input",
        message="Must be positive",
        value=value
    )
```

**Règle:** Problèmes de validation = `ValidationException` (ne pas masquer!)

---

### Pattern 4: Opérations de Stockage

**[Error] AVANT:**
```python
try:
    redis_client.set(key, value)
except Exception as e:
    logger.error(f"Redis error: {e}")
    return False
```

**[OK] APRÈS:**
```python
from api.exceptions import StorageException
import redis.exceptions

try:
    redis_client.set(key, value)
except redis.exceptions.ConnectionError as e:
    raise StorageException(
        storage_type="Redis",
        operation="set",
        message=f"Connection failed: {e}",
        details={"key": key}
    )
except redis.exceptions.TimeoutError as e:
    raise StorageException(
        storage_type="Redis",
        operation="set",
        message=f"Operation timeout: {e}",
        details={"key": key}
    )
except redis.exceptions.RedisError as e:
    raise StorageException(
        storage_type="Redis",
        operation="set",
        message=f"Redis error: {e}",
        details={"key": key}
    )
```

**Règle:** Erreurs de stockage = `StorageException` avec contexte détaillé

---

### Pattern 5: Calculs/Algorithmes

**[Error] AVANT:**
```python
try:
    result = complex_calculation(data)
except Exception as e:
    logger.error(f"Calculation failed: {e}")
    return 0.0  # Valeur par défaut dangereuse!
```

**[OK] APRÈS:**
```python
try:
    result = complex_calculation(data)
except (ValueError, ZeroDivisionError, TypeError) as e:
    # Log avec stacktrace complet pour debugging
    logger.exception(f"Calculation failed with known error: {e}")
    return 0.0  # Fallback acceptable si documenté
except Exception as e:
    # Pour erreurs inattendues: logger.exception + re-raise
    logger.exception(f"Unexpected calculation error: {e}")
    raise  # Re-raise pour alerter
```

**Règle:**
- Erreurs connues = catch spécifiquement + fallback documenté
- Erreurs inconnues = `logger.exception()` + `raise` (NE PAS masquer!)

---

### Pattern 6: Gouvernance/Business Logic

**[Error] AVANT:**
```python
try:
    if not self.validate_policy(policy):
        raise ValueError("Invalid policy")
except Exception as e:
    logger.error(f"Policy validation failed: {e}")
    return default_policy
```

**[OK] APRÈS:**
```python
from api.exceptions import GovernanceException

try:
    if not self.validate_policy(policy):
        raise GovernanceException(
            rule="policy_validation",
            message="Policy does not meet governance requirements",
            details={"policy": policy, "reason": "missing required fields"}
        )
except KeyError as e:
    raise GovernanceException(
        rule="policy_validation",
        message=f"Missing required field: {e}",
        details={"policy": policy}
    )
```

**Règle:** Erreurs de gouvernance = `GovernanceException` avec détails business

---

## Anti-Patterns à Éviter

### Anti-Pattern 1: Silent Failures

```python
try:
    critical_operation()
except Exception:
    pass  # [Error] JAMAIS! Masque tous les bugs
```

**Pourquoi c'est dangereux:**
- Bugs silencieux impossibles à débugger
- État inconsistent non détecté
- Violations de sécurité masquées

### Anti-Pattern 2: Log-and-Swallow

```python
try:
    important_calculation()
except Exception as e:
    logger.error(f"Error: {e}")  #  Log sans re-raise
    return default_value  # Masque le problème
```

**Problème:** L'erreur est loggée mais le bug continue de se propager.

** Solution:** `logger.exception()` + `raise` ou exception spécifique

### Anti-Pattern 3: Bare Except

```python
try:
    operation()
except:  # [Error] Catch même KeyboardInterrupt, SystemExit!
    logger.error("Something failed")
```

**Dangereux:** Capture TOUTES les exceptions y compris les system exits.

---

## Checklist de Migration

Pour chaque fichier:

- [ ] Lire le fichier et identifier les contextes d'usage
- [ ] Pour chaque `except Exception`:
  - [ ] Identifier le type d'opération (API, Storage, Calcul, etc.)
  - [ ] Choisir l'exception appropriée
  - [ ] Remplacer par catch spécifique
  - [ ] Ajouter contexte/détails
  - [ ] Décider: fallback safe OU re-raise
- [ ] Ajouter imports nécessaires en haut du fichier
- [ ] Tester que le fichier compile: `python -m py_compile fichier.py`
- [ ] Vérifier les tests unitaires

---

## Plan de Migration par Fichier

### Priority 1: services/execution/governance.py (42 occurrences)

**Contextes identifiés:**
- Initialisation components (lignes 277, 347) → `ConfigurationException`
- Refresh ML signals (ligne 416) → `APIException`
- Calculs contradiction (ligne 474) → `ValueError` + fallback
- Update state (ligne 595) → `GovernanceException`
- Policy derivation (ligne 698) → `GovernanceException`

**Estimation:** ~2-3h de refactoring minutieux

### Priority 2: services/alerts/alert_storage.py (37 occurrences)

**Contextes:**
- Redis operations → `StorageException`
- Memory fallback → `StorageException`
- Data serialization → `DataException`

**Estimation:** ~1-2h

### Priority 3-5: Autres fichiers critiques

À documenter après migration des 2 premiers.

---

## Testing

Après chaque migration:

```bash
# 1. Vérifier compilation
python -m py_compile services/execution/governance.py

# 2. Run tests unitaires
pytest tests/unit/test_governance.py -v

# 3. Run tests d'intégration
pytest tests/integration/ -k governance -v

# 4. Vérifier logs en dev
# Lancer le serveur et vérifier qu'aucune nouvelle erreur n'apparaît
```

---

## Ressources

- **Custom Exceptions:** `api/exceptions.py`
- **Logging Best Practices:** `docs/LOGGING.md`
- **Python Exception Docs:** https://docs.python.org/3/tutorial/errors.html

---

## Bénéfices Attendus

Après migration complète:

1. **Debugging 10x plus rapide** - Stacktraces précis
2. **Moins de bugs en production** - Erreurs détectées tôt
3. **Meilleure observabilité** - Logs structurés avec contexte
4. **Code plus maintenable** - Intentions claires
5. **Tests plus robustes** - Erreurs spécifiques testables

---

**Note:** Cette migration peut être faite progressivement, fichier par fichier, sans tout refactorer d'un coup.
