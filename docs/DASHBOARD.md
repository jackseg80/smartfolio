# Dashboard

Mise à jour : 27 septembre 2026.

Page : `static/dashboard.html`.

## Répartition Stock Market

En mode **Pro**, sur un écran de taille bureau, la tuile Stock Market affiche un
camembert par classe d'actifs : actions, ETF, cash et autres classes présentes.
Les règles existantes masquent ce graphique en mode Simple et sur mobile.

Le total et le graphique utilisent le même résultat de `fetchSaxoSummary()` :
positions et cash du CSV sélectionné, de l'API Saxo ou de la saisie manuelle.
Le graphique ne lance plus un deuxième chargement avec la clé du sélecteur
utilisée à tort comme nom de fichier. Les valorisations USD sont prioritaires.
Un portefeuille uniquement en cash reste affichable ; le canvas est recréé
si un état vide l'a retiré. Le cache `saxo_summary_v2_` contient les données du
résumé et du graphique, séparées par utilisateur et source.

Global Overview utilise le même fichier CSV Saxo sélectionné et ajoute son
cash après conversion en USD. Le résolveur `resolve_saxo_file_key()` et
`read_saxo_cash()` assurent que la valeur Stocks agrégée correspond à celle de
la tuile Stock Market.

## Export PDF

Le dashboard est capturé à l'échelle 2, puis chaque page est réduite à au plus
6 millions de pixels et encodée en JPEG, qualité 0,88. Ce réglage vise un texte
plus net en zoom tout en gardant un PDF d'une page autour de 1–2 Mo sur un écran
courant. La taille réelle dépend du contenu. Les scripts du dashboard et
l'export utilisent une URL versionnée pour que le navigateur recharge ces
réglages après mise à jour. `imageQuality` et `maxImagePixels` restent réglables.

## Morning Brief

Le bloc a été retiré du dashboard à la demande de l'utilisateur : le résumé
était redondant et imposait la source crypto CoinTracking. Son module n'est
plus chargé par cette page. Ce retrait ne supprime ni le service backend,
ni l'endpoint `/api/morning-brief`, ni sa tâche planifiée.

## Validation

Contrôles automatisés :

```bash
node --experimental-vm-modules --test tests/unit/dashboard_stock_chart.cjs
python -m pytest tests/unit -q --tb=short
```

Sous Windows, activer `.venv/Scripts/Activate.ps1` au préalable. Pour un
checkout de test sans `.env`, définir `ALLOWED_HOSTS=localhost,127.0.0.1,testserver`.
Les tests JavaScript utilisent les dépendances de `package.json`.

Résultats du 27 septembre 2026 : 7 tests JavaScript réussis ; 3 160 tests Python
réussis et 13 ignorés. Rendu Chart.js contrôlé dans Chromium sur une fixture
60 % actions, 30 % ETF, 10 % cash, y compris après un état vide.

Vérification utilisateur après déploiement :

1. Ouvrir le dashboard connecté, recharger sans cache et sélectionner le mode Pro.
2. Choisir le CSV Saxo voulu et vérifier que le total et le camembert correspondent
   à ce fichier, cash inclus. Survoler les secteurs pour lire les montants.
3. Changer de source et vérifier le rafraîchissement des deux affichages.
4. Vérifier que le Morning Brief n'apparaît plus en haut de la page.

La fixture navigateur ne constitue pas une validation du portefeuille réel.

## Déploiement Robot 2 vérifié

Le 27 septembre 2026 à 13:02 CEST, le code `03db2fd1` a été récupéré sur
`main`, reconstruit et activé avec Docker Compose v2. Le conteneur
`smartfolio-api` est `healthy` et `/healthz` retourne HTTP 200 avec `ok: true`.
Les réponses HTTP de `dashboard.html`, `dashboard-main-controller.js` et
`wealth-saxo-summary.js` ont les mêmes empreintes SHA-256 que les fichiers validés.
L'image précédente est conservée sous `smartfolio-prod-rollback:pre-03db2fd1`.
Le contrôle du portefeuille réel par l'utilisateur reste à effectuer.
