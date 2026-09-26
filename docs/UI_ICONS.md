# Icônes et rédaction de l'interface

L'interface utilise **Heroicons 2.2.0, version solid 20 px**. Le sprite SVG et sa
licence MIT sont inclus dans `static/assets/icons/`. Aucun chargement depuis un
CDN n'est nécessaire pour afficher les icônes.

## Utilisation

Dans un module JavaScript :

```javascript
import { renderIcon, setIcon } from '../core/icons.js';

// Icône décorative accompagnant un libellé visible.
button.innerHTML = `${renderIcon('arrow-path')} Refresh`;

// Indicateur autonome avec un libellé accessible.
setIcon(statusElement, 'exclamation-triangle', 'Warning');
```

Dans une page HTML :

```html
<svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20"
     fill="currentColor" aria-hidden="true" focusable="false">
  <use href="/static/assets/icons/heroicons.svg#shield-check"></use>
</svg>
Risk Dashboard
```

- Réutiliser les symboles du sprite ; les nouvelles icônes doivent appartenir au
  même jeu. Mettre à jour la liste autorisée dans `static/core/icons.js` lors d'un ajout.
- Hériter de la couleur du texte avec `currentColor`. Réserver les couleurs de
  statut aux succès, avertissements et erreurs.
- Garder un libellé visible sur les actions. Une action sans texte doit avoir
  un nom accessible ; une icône décorative doit être masquée aux lecteurs d'écran.
- Utiliser du texte simple dans les titres de pages, les listes déroulantes,
  les infobulles et les exports. Ne pas affecter du SVG à `textContent` ou `value`.
- Le composant `empty-state` accepte un nom d'icône, par exemple `icon="inbox"`.
- Les noms et les couleurs des régimes restent la référence. L'ancien export
  `REGIME_EMOJIS` est conservé avec des préfixes vides pour compatibilité.

## Documentation et messages

Les titres et le texte courant n'utilisent pas d'emojis décoratifs. Remplacer
un symbole porteur de sens par un statut explicite : `OK`, `Warning`, `Error`,
`Pending`, ou un libellé précis adapté au contexte. Conserver les avertissements,
les résultats de validation et les états historiques des documents.

Les textes visibles dans l'application restent en anglais. La documentation
technique peut rester en français. Préférer une description factuelle aux
superlatifs et aux formulations promotionnelles.

Source : [Heroicons](https://github.com/tailwindlabs/heroicons/tree/v2.2.0).
Licence : [`static/assets/icons/LICENSE`](../static/assets/icons/LICENSE).
