# Sources V2 - Système Modulaire de Sources

> Architecture plugin pour la gestion des sources de données (crypto, bourse)

## Vue d'ensemble

Le système Sources V2 remplace l'ancienne approche monolithique par une architecture modulaire avec :

- **2 catégories indépendantes** : Crypto et Bourse
- **Mode manuel par défaut** pour les nouveaux utilisateurs
- **1 source exclusive par catégorie** (évite les doublons)
- **Migration automatique** des données existantes
- **Extensible** pour ajouter de nouvelles sources

## Architecture

### Structure des fichiers

```
services/sources/
├── __init__.py              # Exports publics
├── category.py              # Enums (SourceCategory, SourceMode, SourceStatus)
├── base.py                  # SourceBase ABC + BalanceItem dataclass
├── registry.py              # SourceRegistry singleton
├── migration.py             # Migration V1 → V2
├── crypto/
│   ├── manual.py            # ManualCryptoSource (CRUD JSON)
│   ├── cointracking_csv.py  # Wrapper CSV CoinTracking
│   └── cointracking_api.py  # Wrapper API CoinTracking
└── bourse/
    ├── manual.py            # ManualBourseSource (CRUD JSON)
    └── saxobank_csv.py      # Wrapper CSV SaxoBank
```

### Classes principales

#### SourceCategory (Enum)
```python
class SourceCategory(str, Enum):
    CRYPTO = "crypto"   # Cryptomonnaies
    BOURSE = "bourse"   # Actions, ETFs, obligations
```

#### SourceMode (Enum)
```python
class SourceMode(str, Enum):
    MANUAL = "manual"   # Saisie manuelle (défaut)
    CSV = "csv"         # Import fichier
    API = "api"         # Connexion API temps réel
```

#### SourceBase (ABC)
```python
class SourceBase(ABC):
    @classmethod
    @abstractmethod
    def get_source_info(cls) -> SourceInfo: ...

    @abstractmethod
    async def get_balances(self) -> List[BalanceItem]: ...

    @abstractmethod
    async def validate_config(self) -> tuple[bool, Optional[str]]: ...

    @abstractmethod
    def get_status(self) -> SourceStatus: ...
```

#### BalanceItem (Dataclass)
```python
@dataclass
class BalanceItem:
    symbol: str           # BTC, AAPL, etc.
    amount: float         # Quantité
    value_usd: float      # Valeur en USD
    source_id: str        # Source d'origine
    # + champs optionnels: alias, location, price_usd, isin, etc.
```

## API V2 Endpoints

Base URL: `/api/sources/v2`

### Discovery

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/available` | GET | Liste toutes les sources disponibles |
| `/available?category=crypto` | GET | Sources par catégorie |

### Gestion par catégorie

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/{category}/active` | GET | Source active pour la catégorie |
| `/{category}/active` | PUT | Changer la source active |
| `/{category}/status` | GET | Statut de la source active |

### CRUD Manuel Crypto

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/crypto/manual/assets` | GET | Liste les assets manuels |
| `/crypto/manual/assets` | POST | Ajouter un asset |
| `/crypto/manual/assets/{id}` | GET | Détail d'un asset |
| `/crypto/manual/assets/{id}` | PUT | Modifier un asset |
| `/crypto/manual/assets/{id}` | DELETE | Supprimer un asset |

### CRUD Manuel Bourse

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/bourse/manual/positions` | GET | Liste les positions manuelles |
| `/bourse/manual/positions` | POST | Ajouter une position |
| `/bourse/manual/positions/{id}` | GET | Détail d'une position |
| `/bourse/manual/positions/{id}` | PUT | Modifier une position |
| `/bourse/manual/positions/{id}` | DELETE | Supprimer une position |

### Migration

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/migrate` | POST | Migrer un utilisateur vers V2 |
| `/migrate/status` | GET | Statut de migration |

## Configuration utilisateur

### Nouveau format (V2)

```json
// data/users/{user_id}/config.json
{
  "data_source": "category_based",
  "sources": {
    "crypto": {
      "active_source": "manual_crypto",
      "cointracking_csv": { "selected_file": "export.csv" }
    },
    "bourse": {
      "active_source": "saxobank_csv",
      "saxobank_csv": { "selected_file": "positions.csv" }
    }
  }
}
```

### Stockage manuel

```
data/users/{user_id}/
├── manual_crypto/balances.json    # Assets crypto manuels
└── manual_bourse/positions.json   # Positions bourse manuelles
```

## Usage Backend

### Lister les sources disponibles
```python
from services.sources import source_registry, SourceCategory

# Toutes les sources crypto
crypto_sources = source_registry.list_sources(SourceCategory.CRYPTO)

# Sources groupées par mode
by_mode = source_registry.get_sources_by_category(SourceCategory.BOURSE)
```

### Obtenir une source pour un utilisateur
```python
source = source_registry.get_source("manual_crypto", user_id, project_root)
balances = await source.get_balances()
```

### CRUD manuel
```python
from services.sources.crypto.manual import ManualCryptoSource

source = ManualCryptoSource(user_id, project_root)

# Ajouter
asset = source.add_asset(symbol="BTC", amount=0.5, value_usd=25000)

# Lister
assets = source.list_assets()

# Modifier
source.update_asset(asset["id"], amount=0.6)

# Supprimer
source.delete_asset(asset["id"])
```

## Usage Frontend

### Charger le manager
```javascript
import { SourcesManagerV2 } from './sources-manager-v2.js';

const manager = new SourcesManagerV2();
await manager.init();
```

### Changer de source
```javascript
await manager.setActiveSource('crypto', 'cointracking_csv');
```

### CRUD manuel
```javascript
// Ajouter un asset crypto
await manager.addManualAsset('crypto', {
    symbol: 'ETH',
    amount: 2.5,
    value_usd: 5000,
    location: 'Ledger'
});
```

## Rétrocompatibilité

- **Endpoints V1** (`/api/sources/*`) maintenus pendant la transition
- **Config legacy** (`data_source: "cointracking"`) continue de fonctionner
- **Feature flag** `FEATURE_SOURCES_V2=true` pour rollback
- **Données préservées** - migration config seulement, pas de suppression de fichiers

## Ajouter une nouvelle source

1. Créer la classe dans `services/sources/{category}/`
2. Implémenter `SourceBase`
3. Le registry la détecte automatiquement

```python
# services/sources/crypto/binance_api.py
class BinanceAPISource(SourceBase):
    @classmethod
    def get_source_info(cls) -> SourceInfo:
        return SourceInfo(
            id="binance_api",
            name="Binance API",
            category=SourceCategory.CRYPTO,
            mode=SourceMode.API,
            description="Connexion directe Binance",
            icon="api",
            requires_credentials=True,
        )

    async def get_balances(self) -> List[BalanceItem]:
        # Implémentation API Binance
        ...
```

4. Ajouter l'import dans `registry.py`:
```python
try:
    from services.sources.crypto.binance_api import BinanceAPISource
    self.register(BinanceAPISource)
except ImportError as e:
    logger.debug(f"BinanceAPISource not available: {e}")
```

## Tests

```bash
# Unit tests
pytest tests/unit/test_source_registry.py -v
pytest tests/unit/test_manual_sources.py -v

# Integration tests
pytest tests/integration/test_sources_v2.py -v
```

## Voir aussi

- [CLAUDE.md](../CLAUDE.md) - Guide agent principal
- [balance_service.py](../services/balance_service.py) - Intégration balances
- [sources_v2_endpoints.py](../api/sources_v2_endpoints.py) - API endpoints
