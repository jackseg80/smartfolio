# Linting et Formatage de Code Python

> Guide d'utilisation des outils de qualité de code configurés pour SmartFolio

## Outils Configurés

SmartFolio utilise trois outils complémentaires pour garantir la qualité du code Python:

| Outil | Rôle | Configuration |
|-------|------|---------------|
| **black** | Formatage automatique du code | [pyproject.toml](../pyproject.toml) - `[tool.black]` |
| **isort** | Tri et organisation des imports | [pyproject.toml](../pyproject.toml) - `[tool.isort]` |
| **flake8** | Analyse statique (linting) | [.flake8](../.flake8) |

## Usage Rapide

### Script Helper (Recommandé)

Le script `scripts/lint.py` facilite l'utilisation des trois outils:

```bash
# Vérifier le code (mode lecture seule)
python scripts/lint.py --check

# Corriger automatiquement les problèmes
python scripts/lint.py --fix

# Vérifier un fichier/dossier spécifique
python scripts/lint.py --check api/main.py
python scripts/lint.py --fix services/
```

### Commandes Individuelles

```bash
# Activer l'environnement virtuel
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Black - Formatage
black --check api/ services/  # Vérifier
black api/ services/          # Appliquer

# Isort - Imports
isort --check-only api/ services/  # Vérifier
isort api/ services/               # Appliquer

# Flake8 - Linting (lecture seule)
flake8 api/ services/
```

## Configuration

### Black

**Fichier**: `pyproject.toml` - Section `[tool.black]`

- **Line length**: 100 caractères
- **Target**: Python 3.11, 3.12, 3.13
- **Exclusions**: `.venv/`, `archive/`, `data/`, `logs/`, etc.

```toml
[tool.black]
line-length = 100
target-version = ["py311", "py312", "py313"]
```

### Isort

**Fichier**: `pyproject.toml` - Section `[tool.isort]`

- **Profil**: Compatible avec black
- **Line length**: 100 (aligné avec black)
- **Style**: Trailing commas, parenthèses, multi-line

```toml
[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
```

### Flake8

**Fichier**: `.flake8`

- **Max line length**: 100
- **Max complexity** (McCabe): 15
- **Ignores**: E203 (black compat), W503 (black style), E501 (handled by black)

```ini
[flake8]
max-line-length = 100
max-complexity = 15
extend-ignore = E203, W503, E501
```

## Workflow Recommandé

### Avant Commit

```bash
# 1. Vérifier
python scripts/lint.py --check

# 2. Si erreurs, corriger automatiquement
python scripts/lint.py --fix

# 3. Re-vérifier
python scripts/lint.py --check

# 4. Commit
git add .
git commit -m "..."
```

### Pre-commit Hook (Optionnel)

Pour automatiser les vérifications:

```yaml
# .pre-commit-config.yaml (à créer)
repos:
  - repo: https://github.com/psf/black
    rev: 24.0.0
    hooks: [black]
  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks: [isort]
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks: [flake8]
```

Installer: `pip install pre-commit && pre-commit install`

## Erreurs Fréquentes

### F401 - Import inutilisé

```python
# ❌ Avant
from typing import List, Dict
def foo(): return {}  # List non utilisé

# ✅ Après
from typing import Dict
def foo(): return {}
```

### E501 - Ligne trop longue

Black gère automatiquement, mais si impossible:

```python
# ❌ Avant
result = some_very_long_function_name(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)

# ✅ Après (black auto)
result = some_very_long_function_name(
    arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
)
```

### Imports mal triés

```python
# ❌ Avant
from services.portfolio import get_portfolio
import os
from typing import Dict

# ✅ Après (isort auto)
import os
from typing import Dict

from services.portfolio import get_portfolio
```

## Intégration CI/CD

Ajouter au workflow GitHub Actions / GitLab CI:

```yaml
- name: Lint Python
  run: |
    pip install black isort flake8
    python scripts/lint.py --check
```

## Gradual Adoption

**Stratégie**: Application progressive sur le codebase existant (434 fichiers Python)

1. **Phase 1**: Nouveaux fichiers uniquement
2. **Phase 2**: Fichiers modifiés
3. **Phase 3**: Modules critiques (api/, services/)
4. **Phase 4**: Codebase complet

**Raison**: Éviter le "Big Bang" - 434 fichiers nécessitent review manuelle post-formatage.

## Ressources

- [Black Documentation](https://black.readthedocs.io/)
- [Isort Documentation](https://pycqa.github.io/isort/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)

---

**Dernière mise à jour**: 2026-01-29
**Version**: SmartFolio 2.0.0
