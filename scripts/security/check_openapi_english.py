"""Fail when user-visible OpenAPI text contains known French terms."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.main import app


FRENCH_TERMS = re.compile(
    r"\b("
    r"retourne|récupère|utilisateur|données|alerte|portefeuille|exécute|génère|"
    r"calcule|permet|supprime|modifie|ajoute|liste|historique|"
    r"recommandation|enrichit|statistiques|système|déclenche|assigne|"
    r"mot de passe|requis|optionnel|aucun|erreur|méthode|scénario|chocs|"
    r"espéré|répartition|volatilité|stratégie|résultats|ordres|début|"
    r"utilisées|décision|contexte|humaine|amélioration|différemment|"
    r"parallèle|tendance|régime|métriques|largeur|marché|confiance|"
    r"détection|signaux|cibles|suggérées|génération|canal|activé|"
    r"spécifique|personnalisé|trouvée|explications|appliquée|forcer|"
    r"récent|fichiers|chemins|relatifs|fichier|modèle|entraînement|"
    r"prédit|écart-type|borne|inférieure|supérieure|prédiction|demandée|"
    r"qualité|succès|agrégées|échec|réponses|requête|détail|état|santé|"
    r"approuver|rejeter|additionnelles"
    r")\b",
    re.IGNORECASE,
)


def _find_violations(node, location: str = "openapi") -> list[str]:
    violations: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child_location = f"{location}.{key}"
            if (
                key in {"summary", "description", "title"}
                and isinstance(value, str)
                and FRENCH_TERMS.search(value)
            ):
                violations.append(f"{child_location}: {value[:180]!r}")
            violations.extend(_find_violations(value, child_location))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            violations.extend(_find_violations(value, f"{location}[{index}]"))
    return violations


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    schema = app.openapi()
    violations = _find_violations(schema)

    if violations:
        print("French user-visible OpenAPI text detected:")
        print("\n".join(violations))
        return 1
    print(f"OpenAPI English check passed ({len(schema.get('paths', {}))} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
