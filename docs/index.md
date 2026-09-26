# Portail de la Documentation

Bienvenue dans le centre de documentation de **SmartFolio**. Ce portail centralise l'état de l'art du système, ses règles critiques et ses guides d'utilisation.

---

## Essentiels
Accès rapide pour démarrer et comprendre le projet.

- **[Guide Développeur](developer.md)** — Installation, setup et conventions.
- **[Guide Utilisateur](user-guide.md)** — Comment utiliser l'interface et comprendre les métriques.
- **[Règles Critiques (Agent IA)](../CLAUDE.md)**  — **Source canonique** pour les développeurs et agents IA.
- **[Dépannage](troubleshooting.md)** — Solutions aux problèmes courants.

---

## Architecture & Design
La structure technique et les standards du projet.

- **[Architecture Système](architecture.md)** — Design global, flux de données et composants.
- **[Référence API](API_REFERENCE.md)** — Documentation technique de tous les endpoints.
- **[Cartographie des Modules](MODULE_MAP.md)** — Organisation des fichiers JavaScript et services Python.
- **[Standards UI/UX](UX_GUIDE.md)** — Principes de design et composants Web.
- **[Icônes et rédaction](UI_ICONS.md)** — Jeu SVG unifié et conventions de présentation.
- **[Système de Télémétrie](LOGGING.md)** — Gestion des logs et monitoring système.

---

## Fonctionnalités Clés
Spécifications détaillées des modules métiers.

### Intelligence Artificielle & Insights
- **[Assistant IA Global](AI_CHAT_GLOBAL.md)** — Système d'aide contextuel unifié (Multi-provider).
- **[Moteur de Phase](PHASE_ENGINE.md)** — Détection automatique des cycles de marché.
- **[Index de Décision V2](DECISION_INDEX_V2.md)** — Système dual de scoring (Régime vs Qualité).

### Gestion de Portefeuille & Allocation
- **[Moteur d'Allocation V2](ALLOCATION_ENGINE_V2.md)** — Stratégie hiérarchique et gestion des "floors".
- **[Optimisation de Portefeuille](PORTFOLIO_OPTIMIZATION_GUIDE.md)** — Guide des 6 algorithmes de Markowitz.
- **[Système de Sources](SOURCES_SYSTEM.md)** — Résolution multi-source (CSV, API, Banks).

### Risque & Protection
- **[Sémantique du Risque](RISK_SEMANTICS.md)**  — **Convention cruciale** sur le calcul des scores.
- **[Stop Loss Intelligent](STOP_LOSS_SYSTEM.md)** — Méthodes adaptatives et trailing stops.
- **[Stress Testing & Monte Carlo](STRESS_TESTING_MONTE_CARLO.md)** — Simulations de crise et probabilités.

### Bourse & Patrimoine
- **[Scanner d'Opportunités](MARKET_OPPORTUNITIES_SYSTEM.md)** — Détection de gaps sectoriels mondiaux.
- **[Module Patrimoine](PATRIMOINE_MODULE.md)** — Suivi unifié des actifs (Banks, Immobilier, Cash).
- **[Intégration Saxo](SAXO_OAUTH2_INTEGRATION.md)** — Connexion OAuth2 et gestion des comptes titres.

---

## Développement & Qualité
Outils et procédures pour maintenir la robustesse du code.

- **[Guide du Développeur](developer.md)** — Setup local, environnement et conventions.
- **[Workflow de Contribution](../CONTRIBUTING.md)** — Règles Git, commits et Pull Requests.
- **[Guide des Tests](TESTING_GUIDE.md)** — Stratégie de tests unitaires, intégration et E2E.
- **[Dette Technique](TECHNICAL_DEBT.md)** — Suivi des points d'amélioration et bugs connus.

---

## Archives
Historique du projet et notes de travail.

- **[Logs de Développement](_archive/2025_dev_logs/)** — Historique des correctifs et sessions de 2025.
- **[Rapports d'Audit](_archive/audit_reports/)** — Anciens audits de sécurité, performance et accessibilité.
- **[Notes de Session](_archive/session_notes/)** — Détails des anciennes sessions de travail.

---
*Dernière mise à jour : Janvier 2026*
