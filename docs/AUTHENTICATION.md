# Authentication System - JWT

**Status:** ✅ Production Ready (Dec 2025)

Système d'authentification JWT complet pour SmartFolio avec gestion sécurisée des passwords et sessions.

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage Frontend](#usage-frontend)
6. [Usage Backend](#usage-backend)
7. [Gestion des Users](#gestion-des-users)
8. [Sécurité](#sécurité)
9. [Migration & Compatibilité](#migration--compatibilité)

---

## 🎯 Vue d'ensemble

### Features

- ✅ **JWT Tokens** avec expiration configurable (7 jours par défaut)
- ✅ **Password Hashing** avec bcrypt (cost factor 12)
- ✅ **Multi-utilisateurs** avec isolation des données
- ✅ **RBAC** (Role-Based Access Control)
- ✅ **Auto-logout** sur expiration du token
- ✅ **Dev Mode** bypass pour développement
- ✅ **Compatibilité legacy** (header X-User maintenu)

### Endpoints API

```bash
POST /auth/login       # Login avec username/password → JWT token
POST /auth/logout      # Logout (client-side)
GET  /auth/verify      # Vérifier validité d'un token
```

---

## 🏗️ Architecture

### Backend

```
api/
  auth_router.py       # Endpoints login/logout/verify
  deps.py              # Dependencies JWT (get_current_user_jwt, require_admin_role_jwt)

config/
  users.json           # User registry avec password_hash

scripts/
  setup_passwords.py   # Script génération passwords
```

### Frontend

```
static/
  login.html                    # Page de login
  core/
    auth-guard.js              # Module protection auth
  components/
    nav.js                     # Navigation avec bouton logout
```

### Flow d'Authentification

```
1. User → Login (username/password)
   ↓
2. Backend vérifie password_hash (bcrypt)
   ↓
3. Backend génère JWT token (exp: 7 jours)
   ↓
4. Frontend stocke token dans localStorage
   ↓
5. Toutes les requêtes incluent: Authorization: Bearer {token}
   ↓
6. Backend valide token à chaque requête (deps.py)
   ↓
7. Token expiré → Auto-redirect vers login
```

---

## 📦 Installation

### 1. Installer les dépendances

```bash
pip install passlib[bcrypt] python-jose[cryptography]
```

**Déjà ajouté dans `requirements.txt` :**
```txt
passlib[bcrypt]>=1.7.4
python-jose[cryptography]>=3.3.0
```

### 2. Générer les passwords

```bash
# Générer passwords pour tous les users
python scripts/setup_passwords.py

# Définir password pour un user spécifique
python scripts/setup_passwords.py --user jack --password "MySecurePassword123!"

# Regénérer tous les passwords (force)
python scripts/setup_passwords.py --force
```

**Output exemple :**
```
============================================================
SmartFolio - Password Setup
============================================================

✅ Password generated for 'demo' (Démo)
✅ Password generated for 'jack' (Jack)
✅ Users config saved to config/users.json

============================================================
Password Summary - SAVE THESE CREDENTIALS SECURELY
============================================================

User: Démo (demo)
Password: aB3!xK9mZp2@Qw5Y
Roles: viewer
------------------------------------------------------------
User: Jack (jack)
Password: Pz8$Lm4!Nq1@Rj7T
Roles: admin, ml_admin, governance_admin
------------------------------------------------------------

⚠️  WARNING: Save these passwords now! They cannot be retrieved later.
✅ Setup complete. 2 password(s) configured.
```

### 3. Configuration JWT Secret (optionnel)

**Fichier `.env` :**
```bash
# JWT Configuration
JWT_SECRET_KEY=your-super-secret-key-min-32-characters
DEV_SKIP_AUTH=0  # 1 pour bypass auth en dev (non recommandé)
```

**Génération d'un secret sécurisé :**
```python
import secrets
print(secrets.token_urlsafe(32))
# → "x4Kz9mNpQ2aB5cD8eF1gH3jL6nO9rS2tU5wX8yZ1"
```

---

## ⚙️ Configuration

### Structure `users.json`

```json
{
  "default": "demo",
  "roles": {
    "admin": "Full system access",
    "ml_admin": "ML model training",
    "governance_admin": "Execution & governance",
    "viewer": "Read-only access"
  },
  "users": [
    {
      "id": "jack",
      "label": "Jack",
      "password_hash": "$2b$12$...",
      "roles": ["admin", "ml_admin"],
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

**Champs :**
- `id` : Identifiant unique (lowercase)
- `label` : Nom affiché
- `password_hash` : Hash bcrypt du password
- `roles` : Array de rôles RBAC
- `status` : `"active"` ou `"inactive"`
- `created_at` : Timestamp création

---

## 🖥️ Usage Frontend

### Protection d'une page HTML

```html
<!DOCTYPE html>
<html>
<head>
    <title>Protected Page</title>
</head>
<body>
    <!-- Page content -->

    <script type="module">
        import { checkAuth, getAuthHeaders } from './core/auth-guard.js';

        // Vérifier auth au chargement
        await checkAuth();

        // Faire des requêtes authentifiées
        const response = await fetch('/api/portfolio/metrics', {
            headers: getAuthHeaders()
        });
    </script>
</body>
</html>
```

### Module `auth-guard.js`

**Fonctions disponibles :**

```javascript
import {
    checkAuth,           // Vérifier auth + redirect si nécessaire
    logout,              // Déconnexion + redirect login
    getAuthHeaders,      // Headers pour fetch (Authorization + X-User)
    getAuthToken,        // Récupérer JWT token
    getCurrentUser,      // Récupérer user_id
    getUserInfo,         // Récupérer user info complète
    hasRole,             // Vérifier si user a un rôle
    isAdmin,             // Vérifier si user est admin
    requireRole          // Require role ou redirect
} from './core/auth-guard.js';

// Exemple: Vérifier au chargement
await checkAuth();

// Exemple: Fetch authentifié
const data = await fetch('/api/endpoint', {
    headers: getAuthHeaders()
});

// Exemple: Vérifier rôle admin
if (isAdmin()) {
    console.log('User is admin');
}

// Exemple: Require admin ou redirect
requireRole('admin', 'Cette page nécessite les droits admin');
```

### Bouton Logout (nav.js)

Le bouton logout est déjà intégré dans `nav.js` :

```javascript
// Click sur bouton logout
const authGuard = await import('./core/auth-guard.js');
await authGuard.logout(true);  // true = afficher message
```

---

## 🔧 Usage Backend

### Endpoints protégés (JWT)

```python
from fastapi import Depends
from api.deps import get_current_user_jwt

@router.get("/protected-endpoint")
async def protected_endpoint(user: str = Depends(get_current_user_jwt)):
    # user contient l'user_id extrait du JWT
    return {"message": f"Hello {user}"}
```

### Endpoints admin (JWT + RBAC)

```python
from api.deps import require_admin_role_jwt

@router.get("/admin/users")
async def list_users(user: str = Depends(require_admin_role_jwt)):
    # user est garanti avoir le rôle "admin"
    return {"users": [...]}
```

### Compatibilité Legacy (X-User header)

```python
from api.deps import get_active_user

@router.get("/legacy-endpoint")
async def legacy_endpoint(user: str = Depends(get_active_user)):
    # Supporte à la fois JWT et X-User header (fallback)
    return {"user": user}
```

### Mode DEV Bypass

```python
# Dans .env
DEV_SKIP_AUTH=1

# → Toutes les fonctions `get_current_user_jwt()` retournent "demo"
# → Utile pour développement rapide
```

---

## 👥 Gestion des Users

### Créer un nouveau user

1. **Modifier `config/users.json` :**

```json
{
  "users": [
    {
      "id": "nouveau_user",
      "label": "Nouveau User",
      "roles": ["viewer"],
      "status": "active",
      "created_at": "2025-12-28T12:00:00Z"
    }
  ]
}
```

2. **Générer le password :**

```bash
python scripts/setup_passwords.py --user nouveau_user --password "PasswordTemporaire123!"
```

3. **Partager les credentials de manière sécurisée**

### Réinitialiser un password

```bash
python scripts/setup_passwords.py --user jack --password "NewPassword123!" --force
```

### Désactiver un user

Modifier `users.json` :
```json
{
  "id": "user_a_desactiver",
  "status": "inactive"
}
```

Le user ne pourra plus se connecter (même avec un token valide).

### Supprimer un user

1. Retirer de `users.json`
2. Supprimer le dossier `data/users/{user_id}/`
3. Les tokens existants deviendront invalides automatiquement

---

## 🔒 Sécurité

### Password Hashing

**Bcrypt avec cost factor 12 :**
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
password_hash = pwd_context.hash("plain_password")
```

**Pourquoi bcrypt ?**
- ✅ Résistant aux attaques brute-force (slow hashing)
- ✅ Salt automatique
- ✅ Cost factor ajustable (futureproof)

### JWT Token

**Payload JWT :**
```json
{
  "sub": "jack",                    // User ID
  "roles": ["admin", "ml_admin"],   // Rôles RBAC
  "label": "Jack",                  // Nom affiché
  "exp": 1735468800,                // Expiration timestamp
  "iat": 1734864000                 // Issued at timestamp
}
```

**Validations :**
- ✅ Signature cryptographique (HMAC-SHA256)
- ✅ Expiration automatique (7 jours)
- ✅ Vérification user existe toujours
- ✅ Vérification user status = "active"

### Bonnes Pratiques

1. **Passwords sécurisés :**
   - Min 12 caractères
   - Mix lettres/chiffres/caractères spéciaux
   - Pas de mots du dictionnaire

2. **JWT Secret :**
   - Min 32 caractères
   - Généré aléatoirement
   - Jamais commité dans Git

3. **HTTPS en production :**
   - Tokens transmis uniquement via HTTPS
   - Cookie `Secure` flag si cookies utilisés

4. **Logs d'audit :**
   - Login succès/échec loggés
   - Admin access loggé
   - Token expiration loggée

---

## 🔄 Migration & Compatibilité

### Endpoints existants (X-User)

**Les endpoints existants continuent de fonctionner avec X-User :**

```javascript
// Frontend legacy (continue de fonctionner)
fetch('/api/portfolio/metrics', {
    headers: { 'X-User': 'jack' }
});
```

**Backend supporte les deux méthodes :**
```python
from api.deps import get_active_user  # Supporte X-User ET JWT

@router.get("/endpoint")
async def endpoint(user: str = Depends(get_active_user)):
    # Fonctionne avec:
    # - Header "X-User: jack" (legacy)
    # - Header "Authorization: Bearer {token}" (nouveau JWT)
```

### Migration Progressive

**Phase 1 (Actuelle) - Dual Mode :**
- ✅ JWT tokens générés au login
- ✅ Endpoints acceptent JWT OU X-User
- ✅ Frontend envoie les deux headers

**Phase 2 (Future) - JWT Obligatoire :**
- Remplacer `get_active_user` → `get_current_user_jwt`
- Retirer support X-User
- Forcer login pour tous

### Frontend Hybrid Headers

```javascript
import { getAuthHeaders } from './core/auth-guard.js';

// Retourne automatiquement:
const headers = getAuthHeaders();
// {
//   "Authorization": "Bearer eyJhbGc...",  // JWT (prioritaire)
//   "X-User": "jack"                       // Fallback legacy
// }
```

---

## 🧪 Tests

### Test Login

```bash
curl -X POST "http://localhost:8080/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=jack&password=YourPassword123"

# Response:
{
  "ok": true,
  "data": {
    "token": "eyJhbGc...",
    "token_type": "bearer",
    "expires_in": 604800,
    "user": {
      "id": "jack",
      "label": "Jack",
      "roles": ["admin"]
    }
  }
}
```

### Test Verify Token

```bash
curl "http://localhost:8080/auth/verify?token=eyJhbGc..."

# Response:
{
  "ok": true,
  "data": {
    "valid": true,
    "user_id": "jack",
    "roles": ["admin"],
    "expires_at": "2025-01-04T12:00:00"
  }
}
```

### Test Protected Endpoint

```bash
curl "http://localhost:8080/api/portfolio/metrics" \
  -H "Authorization: Bearer eyJhbGc..."

# Response: données portfolio (si token valide)
```

---

## 📚 Ressources

### Documentation Liée

- [`CLAUDE.md`](../CLAUDE.md) - Guide agent complet
- [`ADMIN_DASHBOARD.md`](ADMIN_DASHBOARD.md) - Admin Dashboard RBAC

### Librairies Utilisées

- [passlib](https://passlib.readthedocs.io/) - Password hashing
- [python-jose](https://python-jose.readthedocs.io/) - JWT tokens
- [bcrypt](https://github.com/pyca/bcrypt/) - Bcrypt backend

### Standards

- [RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519) - JWT Specification
- [OWASP Authentication](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html) - Best practices

---

## 🐛 Troubleshooting

### "Invalid or expired token"

**Cause :** Token JWT expiré ou invalide

**Solution :**
1. Vérifier expiration du token (7 jours par défaut)
2. Se reconnecter via `/static/login.html`
3. Vérifier `JWT_SECRET_KEY` n'a pas changé

### "User not found"

**Cause :** User supprimé de `users.json` mais token encore valide

**Solution :**
1. Vérifier user existe dans `config/users.json`
2. Vérifier `status: "active"`

### "Admin role required"

**Cause :** User n'a pas le rôle `"admin"`

**Solution :**
1. Vérifier `roles` dans `users.json`
2. Ajouter `"admin"` au tableau `roles`

### "Password verification failed"

**Cause :** Password incorrect

**Solution :**
1. Vérifier le password saisi
2. Réinitialiser avec `scripts/setup_passwords.py --user X --password Y --force`

---

**Documentation générée:** Dec 2025
**Version:** SmartFolio v2.0
**Status:** ✅ Production Ready
