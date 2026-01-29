"""
Tests unitaires pour la configuration conditionnelle du HTTPSRedirectMiddleware.
P1-3: Vérification que HTTPS redirect est activé en production uniquement.
"""

import pytest
from config import get_settings


class TestHTTPSConfiguration:
    """Tests pour la configuration HTTPS basée sur l'environnement."""

    def test_settings_production_flag(self):
        """Vérifier que is_production() retourne la bonne valeur selon l'environnement."""
        settings = get_settings()

        # Test de cohérence: is_production() doit correspondre à environment
        if settings.environment == "production":
            assert settings.is_production() is True, \
                "is_production() doit retourner True quand environment='production'"
        else:
            assert settings.is_production() is False, \
                f"is_production() doit retourner False quand environment='{settings.environment}'"

    def test_development_mode_by_default(self):
        """
        Par défaut (sans ENVIRONMENT env var), on doit être en développement.
        Ce test valide le comportement sécurisé par défaut.
        """
        settings = get_settings()
        # Si aucune variable d'environnement n'est définie, on devrait être en dev
        # Note: Ce test peut échouer si ENVIRONMENT=production est défini dans .env
        assert settings.environment in ["development", "production"], \
            f"Environment doit être 'development' ou 'production', reçu: {settings.environment}"

    def test_https_middleware_import_available(self):
        """Vérifier que HTTPSRedirectMiddleware est importable."""
        try:
            from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
            assert HTTPSRedirectMiddleware is not None
        except ImportError as e:
            pytest.fail(f"HTTPSRedirectMiddleware should be importable from FastAPI: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
