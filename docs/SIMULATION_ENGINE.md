# Simulation Engine

> **Note** : Document consolidé depuis `SIMULATION_ENGINE_ALIGNMENT.md` (266 lignes).
> Sections 1–2 reprises intégralement, sections 3–5 condensées.

---

## 1. Objectif

Le moteur de simulation permet d'exécuter des scénarios "what-if" sur le portefeuille
en utilisant les mêmes règles de Risk, Cycle et Governance que le système de production.

Caractéristiques :
- Multi-tenant (chaque simulation isolée par `user_id`)
- Historique indépendant (`_sim` vs `_prod`)
- Injection des politiques Governance et Risk
- Support de presets (10 profils disponibles dans `simulations.html`)

---

## 2. Architecture

- **Entrée** : snapshot du `risk-dashboard-store`
- **Core** : `static/modules/simulation-engine.js`
- **Sortie** : résultats consolidés (risk, DI, caps déclenchés)
- **UI** : `simulations.html` + composants panels

Modules clés :
- `simulation-engine.js` : logique centrale
- `risk-dashboard-store.js` : store partagé Risk/Governance
- `unified-insights-v2.js` : réutilisé pour DI calculé
- `selectors/governance.js` : sélection TTL/staleness

---

## 3. Fonctionnalités (Résumé)

- **Injection Governance** : capacité de tester des caps journaliers, hebdo, mensuels
- **Risk Alignment** : DI simulé utilise la même règle Risk [0..100] positive
- **Presets** : 10 scénarios intégrés (conservateur, agressif, cycle bull/bear)
- **Contradiction** : simulation calcule aussi l'indice de contradiction
- **UI** : affichage badge + caps déclenchés

---

## 4. Alignement avec Production (Condensé)

- Les mêmes fonctions backend sont appelées (`/api/risk/*`, `/api/ml/*`)
- Les stores `_sim` et `_prod` sont strictement séparés
- Les ETag et timestamps timezone Europe/Zurich garantissent cohérence
- Une simulation ne modifie jamais l'état production

---

## 5. QA Checklist

- [OK] DI simulé = DI prod avec mêmes inputs
- [OK] Pas d'accès direct à API hors sandbox
- [OK] Risk score appliqué positivement (pas de `100 - risk`)
- [OK] Governance caps déclenchés identiques à prod
- [OK] Tests UI : badge + contradiction visibles

---

## Sémantique Risk

> **[Warning] Règle Canonique — Sémantique Risk**
>
> Le **Risk Score** est un indicateur **positif** de robustesse, borné **[0..100]**.
>
> **Convention** : Plus haut = plus robuste (risque perçu plus faible).
>
> **Conséquence** : Dans le Decision Index (DI), Risk contribue **positivement** :
> ```
> DI = wCycle·scoreCycle + wOnchain·scoreOnchain + wRisk·scoreRisk
> ```
>
> ** Interdit** : Ne jamais inverser avec `100 - scoreRisk`.
>
> **Visualisation** : Contribution = `(poids × score) / Σ(poids × score)`
>
>  Source : [docs/RISK_SEMANTICS.md](RISK_SEMANTICS.md)

---

## Référence

- Core : `static/modules/simulation-engine.js`
- Store : `static/core/risk-dashboard-store.js`
- UI : `static/simulations.html`
