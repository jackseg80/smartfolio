# Authentication Setup - Quick Start Guide

Ce guide vous aide à configurer le système d'authentification JWT de SmartFolio.

---

## Prérequis

- Python 3.9+
- SmartFolio backend installé
- Dépendances JWT installées (voir étape 1)

---

## Installation en 5 Étapes

### Installer les dépendances JWT

```bash
pip install passlib[bcrypt] python-jose[cryptography]
```

**Ou via requirements.txt :**
```bash
pip install -r requirements.txt
```

### Configurer le JWT Secret

**Créer ou modifier `.env` :**
```bash
# Copier le template
cp .env.example .env

# Générer un secret sécurisé
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Ajouter dans `.env` :**
```bash
# JWT Configuration
JWT_SECRET_KEY=votre-secret-genere-ici
DEV_SKIP_AUTH=0  # 0 = auth activée, 1 = bypass (dev only)
```

### Générer les passwords utilisateurs

```bash
# Générer passwords pour tous les users
python scripts/setup_passwords.py
```

**Output exemple :**
```
============================================================
SmartFolio - Password Setup
============================================================

[OK] Password generated for 'demo' (Démo)
[OK] Password generated for 'jack' (Jack)

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

  WARNING: Save these passwords now! They cannot be retrieved later.
```

** Astuce:** Sauvegardez ces passwords dans un gestionnaire de mots de passe (1Password, Bitwarden, etc.)

### (Optionnel) Protéger les pages HTML

```bash
# Preview (dry-run)
python scripts/add_auth_guards.py

# Appliquer les protections
python scripts/add_auth_guards.py --apply
```

**Protège automatiquement :**
- dashboard.html
- analytics-unified.html
- risk-dashboard.html
- saxo-dashboard.html
- admin-dashboard.html
- ... toutes les pages principales

### Démarrer le serveur

```bash
# Activer l'environnement virtuel
.venv\Scripts\Activate.ps1  # Windows PowerShell
# ou
source .venv/bin/activate   # Linux/Mac

# Démarrer le serveur
python -m uvicorn api.main:app --port 8080
```

---

## Premier Login

1. **Ouvrir le navigateur :** `http://localhost:8080/static/login.html`

2. **Se connecter avec les credentials générés :**
   - **Username:** `jack` (admin) ou `demo` (viewer)
   - **Password:** Le password affiché par `setup_passwords.py`

3. **Après login réussi :**
   - Redirect automatique vers `/static/dashboard.html`
   - Token JWT stocké dans `localStorage` (valide 7 jours)
   - Bouton "Logout" visible dans la navigation

---

## Commandes Utiles

### Générer un password pour un user spécifique

```bash
python scripts/setup_passwords.py --user jack --password "MonSuperPassword123!"
```

### Regénérer tous les passwords (force)

```bash
python scripts/setup_passwords.py --force
```

### Vérifier qu'un password fonctionne

```bash
curl -X POST "http://localhost:8080/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=jack&password=VotrePassword"
```

**Response attendue :**
```json
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

### Bypass auth en mode DEV (dev uniquement!)

**Dans `.env` :**
```bash
DEV_SKIP_AUTH=1  # Bypass auth (toutes les pages accessibles sans login)
```

**[Warning] Attention:** Ne JAMAIS activer en production !

---

## Workflow Complet

### Premier Démarrage

```bash
# 1. Installer dépendances
pip install -r requirements.txt

# 2. Générer JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
# → Copier dans .env: JWT_SECRET_KEY=...

# 3. Générer passwords
python scripts/setup_passwords.py
# → Sauvegarder les passwords affichés

# 4. Protéger les pages HTML (optionnel)
python scripts/add_auth_guards.py --apply

# 5. Démarrer serveur
python -m uvicorn api.main:app --port 8080

# 6. Se connecter
# Naviguer vers: http://localhost:8080/static/login.html
```

### Ajout d'un nouvel utilisateur

```bash
# 1. Éditer config/users.json
# Ajouter:
{
  "id": "nouveau_user",
  "label": "Nouveau User",
  "roles": ["viewer"],
  "status": "active",
  "created_at": "2025-12-28T12:00:00Z"
}

# 2. Générer son password
python scripts/setup_passwords.py --user nouveau_user --password "PasswordTemporaire123!"

# 3. Partager les credentials de manière sécurisée
```

### Réinitialiser un password oublié

```bash
python scripts/setup_passwords.py --user jack --password "NouveauPassword123!" --force
```

---

## Documentation Complète

- **Guide Complet:** [`docs/AUTHENTICATION.md`](docs/AUTHENTICATION.md)
- **Architecture Système:** Endpoints, JWT flow, sécurité
- **Usage Frontend:** `auth-guard.js` API
- **Usage Backend:** Dependencies, RBAC, protection
- **Troubleshooting:** Erreurs communes et solutions

---

## FAQ

### Q: Où sont stockés les passwords ?

**A:** Hashés avec bcrypt dans `config/users.json` (champ `password_hash`). Les passwords en clair ne sont jamais stockés.

### Q: Combien de temps dure un token JWT ?

**A:** 7 jours par défaut. Configurable dans `api/auth_router.py` (variable `ACCESS_TOKEN_EXPIRE_DAYS`).

### Q: Comment désactiver l'authentification pour le dev ?

**A:** Ajouter `DEV_SKIP_AUTH=1` dans `.env`. **Attention:** Ne JAMAIS activer en production !

### Q: Que se passe-t-il si le token expire ?

**A:** Redirect automatique vers `/static/login.html` avec message "session_expired".

### Q: Les endpoints existants (X-User) fonctionnent encore ?

**A:** Oui ! Le système est rétrocompatible. Les endpoints acceptent à la fois JWT et X-User header.

### Q: Comment créer un admin ?

**A:** Éditer `config/users.json` et ajouter `"admin"` dans le tableau `roles`.

---

## Sécurité - Important !

[OK] **À FAIRE :**
- Changer `JWT_SECRET_KEY` en production
- Utiliser des passwords forts (min 12 caractères)
- Activer HTTPS en production
- Sauvegarder les passwords de manière sécurisée

[Error] **NE JAMAIS :**
- Committer `.env` dans Git
- Partager passwords par email/chat
- Activer `DEV_SKIP_AUTH=1` en production
- Réutiliser le même password pour plusieurs users

---

## Aide & Support

**Problème de login ?**
- Vérifier que `password_hash` existe dans `config/users.json`
- Vérifier que `status: "active"`
- Vérifier les logs serveur : `logs/app.log`

**Token invalide/expiré ?**
- Se reconnecter via `/static/login.html`
- Vérifier que `JWT_SECRET_KEY` n'a pas changé

**Page bloquée ?**
- Vérifier que `checkAuth()` est appelé dans le script
- Vérifier que le token existe : `localStorage.getItem('authToken')`

---

**Documentation générée:** Dec 2025
**Version:** SmartFolio v2.0
**Status:** [OK] Production Ready
