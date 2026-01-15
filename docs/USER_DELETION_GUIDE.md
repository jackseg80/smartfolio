# Guide de Suppression des Utilisateurs

## Vue d'ensemble

SmartFolio propose maintenant **deux types de suppression** d'utilisateurs :
- **Soft Delete** (par défaut, recommandé)
- **Hard Delete** (suppression permanente)

---

## Soft Delete (Défaut)

### Comportement
1. Marque l'utilisateur comme `status: "inactive"` dans `config/users.json`
2. Renomme le dossier utilisateur : `data/users/{user_id}` → `data/users/{user_id}_deleted_{timestamp}`
3. L'utilisateur **reste dans users.json**

### Avantages
✅ Réversible manuellement si besoin
✅ Garde les données pour audit
✅ Empêche l'accès mais préserve l'historique

### Inconvénients
❌ Impossible de recréer un utilisateur avec le même ID
❌ L'utilisateur reste visible (status inactive)

### Utilisation

**Via Admin Dashboard:**
1. Aller dans Admin Dashboard → User Management
2. Cliquer sur "🗑️ Delete" sur l'utilisateur
3. **Choisir "Soft Delete (Recommended)"**
4. Confirmer

**Via API:**
```bash
curl -X DELETE "http://localhost:8080/admin/users/{user_id}" \
  -H "X-User: jack"
```

**Via PowerShell:**
```powershell
curl.exe -X DELETE "http://localhost:8080/admin/users/toto" -H "X-User: jack"
```

---

## Hard Delete (Permanent)

### Comportement
1. **Supprime complètement** l'utilisateur de `config/users.json`
2. **Supprime le dossier** `data/users/{user_id}` et toutes ses données
3. L'utilisateur **disparaît complètement** du système

### Avantages
✅ Permet de recréer un utilisateur avec le même ID
✅ Nettoyage complet du système
✅ Libère l'espace disque

### Inconvénients
❌ **IRRÉVERSIBLE** - aucune restauration possible
❌ Perte de toutes les données utilisateur
❌ Risque d'erreur si mauvais utilisateur sélectionné

### ⚠️ À utiliser uniquement si :
- Vous devez recréer un utilisateur avec le même ID
- Vous êtes sûr à 100% de vouloir supprimer définitivement
- Les données ne sont plus nécessaires

### Utilisation

**Via Admin Dashboard:**
1. Aller dans Admin Dashboard → User Management
2. Cliquer sur "🗑️ Delete" sur l'utilisateur
3. **⚠️ Choisir "Hard Delete (Permanent)"**
4. Confirmer (bouton rouge)

**Via API:**
```bash
curl -X DELETE "http://localhost:8080/admin/users/{user_id}?hard_delete=true" \
  -H "X-User: jack"
```

**Via PowerShell:**
```powershell
curl.exe -X DELETE "http://localhost:8080/admin/users/toto?hard_delete=true" -H "X-User: jack"
```

---

## Tableau Comparatif

| Critère | Soft Delete | Hard Delete |
|---------|-------------|-------------|
| **Réversible** | ✅ Manuellement | ❌ Non |
| **Données préservées** | ✅ Oui (dossier renommé) | ❌ Supprimées |
| **Présence dans users.json** | ✅ Oui (inactive) | ❌ Non |
| **Recréation possible** | ❌ Non | ✅ Oui |
| **Recommandé** | ✅ Défaut | ⚠️ Cas spécifiques |

---

## Cas d'Usage

### Quand utiliser Soft Delete ?
- Désactivation temporaire d'un compte
- Départ d'un collaborateur (garder l'audit trail)
- Doute sur la suppression
- **Par défaut dans 90% des cas**

### Quand utiliser Hard Delete ?
- Besoin de recréer un utilisateur test avec le même ID
- Compte créé par erreur immédiatement après
- Nettoyage définitif de comptes obsolètes
- **Uniquement si vous êtes certain**

---

## Interface Admin Dashboard

### Modal de Suppression

Lors de la suppression, vous verrez deux options radio :

```
Delete Type:

⚪ Soft Delete (Recommended)
   Mark as inactive and rename data folder. Can be recovered manually if needed.

⚪ ⚠️ Hard Delete (Permanent)
   Remove completely from config and delete data folder. Cannot be undone! User ID can be recreated later.
```

- **Soft Delete** : bordure grise, par défaut
- **Hard Delete** : bordure rouge, avertissement visible

### Messages de Confirmation

**Soft Delete:**
> User "toto" deleted successfully (soft delete)

**Hard Delete:**
> User "toto" deleted permanently

---

## Récupération après Soft Delete

Si vous devez récupérer un utilisateur après soft delete :

1. **Restaurer le dossier:**
   ```bash
   # Retrouver le dossier
   ls data/users/toto_deleted_*

   # Renommer pour restaurer
   mv data/users/toto_deleted_20260115_123456 data/users/toto
   ```

2. **Réactiver dans users.json:**
   ```json
   {
     "id": "toto",
     "status": "inactive"  // Changer en "active"
   }
   ```

3. **Vider le cache:**
   ```bash
   curl -X DELETE "http://localhost:8080/admin/cache/clear?cache_name=users" \
     -H "X-User: jack"
   ```

---

## Sécurité

### Protection contre les suppressions accidentelles

1. **Interdiction de supprimer l'utilisateur par défaut**
   - Le user `default` (généralement "demo") ne peut pas être supprimé

2. **Confirmation requise**
   - Modal de confirmation avant toute suppression

3. **Choix explicite du type**
   - Hard delete nécessite de cocher explicitement l'option rouge

4. **Logs d'audit**
   - Toutes les suppressions sont loguées avec timestamp et admin_user

---

## API Reference

### DELETE /admin/users/{user_id}

**Query Parameters:**
- `hard_delete` (boolean, optional, default: `false`)
  - `false` : Soft delete (défaut)
  - `true` : Hard delete (permanent)

**Headers:**
- `X-User` : Admin user ID (required, must have `admin` role)

**Response (Soft Delete):**
```json
{
  "ok": true,
  "data": {
    "user_id": "toto",
    "deleted": true,
    "delete_type": "soft",
    "deleted_at": "2026-01-15T12:34:56Z",
    "deleted_by": "jack"
  },
  "meta": {
    "message": "User 'toto' deleted successfully (soft (désactivation))"
  }
}
```

**Response (Hard Delete):**
```json
{
  "ok": true,
  "data": {
    "user_id": "toto",
    "deleted": true,
    "delete_type": "hard",
    "deleted_at": "2026-01-15T12:34:56Z",
    "deleted_by": "jack"
  },
  "meta": {
    "message": "User 'toto' deleted successfully (HARD (permanent))"
  }
}
```

---

## Script de Test

Un script PowerShell est disponible pour tester les deux modes :

```powershell
.\scripts\ops\test_user_deletion.ps1
```

Ce script :
1. Crée un utilisateur test
2. Effectue un soft delete
3. Vérifie qu'on ne peut pas recréer
4. Effectue un hard delete
5. Vérifie qu'on peut recréer
6. Nettoie

---

## Changelog

**2026-01-15** (Version 2.0)
- ✅ Ajout du Hard Delete
- ✅ Interface améliorée dans Admin Dashboard avec choix visuel
- ✅ Messages de confirmation distincts
- ✅ Style rouge pour hard delete (avertissement)
- ✅ Réinitialisation automatique sur Soft Delete par sécurité

**Avant** (Version 1.0)
- Uniquement Soft Delete disponible
- Pas de possibilité de recréer un utilisateur avec le même ID
