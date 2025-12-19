# Admin Dashboard - Documentation Système

> **Version:** Phase 1 - Infrastructure RBAC
> **Date:** Décembre 2025
> **Statut:** ✅ Opérationnel (Phase 1 terminée)

## 📋 Vue d'ensemble

Le **Admin Dashboard** est un système d'administration centralisé pour SmartFolio permettant la gestion des utilisateurs, logs, cache, modèles ML et clés API. Il utilise un système RBAC (Role-Based Access Control) pour sécuriser l'accès aux fonctionnalités critiques.

---

## 🎯 Objectifs

1. **Centraliser l'administration** : Un seul point d'accès pour toutes les tâches admin
2. **Sécuriser l'accès** : RBAC avec 4 rôles distincts
3. **Simplifier la gestion** : Interface unifiée, responsive, cohérente avec SmartFolio
4. **Auditer les actions** : Logs complets de toutes les opérations admin

---

## 🔐 Système RBAC

### Rôles Disponibles

| Rôle | Description | Permissions |
|------|-------------|-------------|
| **admin** | Accès complet système | User management, Logs, Cache, ML models, API keys |
| **governance_admin** | Gestion execution & gouvernance | Execution endpoints, Governance rules |
| **ml_admin** | Training & déploiement ML | ML model training, deployment, versioning |
| **viewer** | Lecture seule | Consultation dashboards uniquement |

### Configuration Rôles

**Fichier:** `config/users.json`

```json
{
  "default": "demo",
  "roles": {
    "admin": "Full system access - user management, logs, cache, ML models, API keys",
    "governance_admin": "Execution & governance management",
    "ml_admin": "ML model training & deployment",
    "viewer": "Read-only access"
  },
  "users": [
    {
      "id": "jack",
      "label": "Jack",
      "roles": ["admin", "ml_admin", "governance_admin"],
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Protection Endpoints

**Dependency:** `api/deps.py::require_admin_role()`

```python
from api.deps import require_admin_role

@router.get("/admin/users")
async def list_users(user: str = Depends(require_admin_role)):
    # user est garanti avoir le rôle "admin"
    ...
```

**Comportement :**
- ✅ Vérifie header `X-User` obligatoire
- ✅ Valide que l'user existe dans `config/users.json`
- ✅ Vérifie que l'user a le rôle `"admin"`
- ✅ Logs audit complets (accès granted/denied)
- ✅ Mode dev bypass avec `DEV_OPEN_API=1`

---

## 📁 Architecture Fichiers

### Backend

```
api/
  admin_router.py          # Router principal admin (tous les endpoints)
  deps.py                  # require_admin_role() dependency

services/
  user_management.py       # [Phase 2] User CRUD operations
  log_reader.py            # [Phase 2] Log parsing & filtering
  cache_manager.py         # [Phase 3] Unified cache management
  ml/
    training_executor.py   # [Phase 3] Background ML training jobs
  key_masker.py            # [Phase 4] API key masking utilities

config/
  users.json               # User registry avec rôles RBAC
```

### Frontend

```
static/
  admin-dashboard.html     # Page admin unifiée (6 onglets)

  components/
    nav.js                 # Menu Admin dropdown (lignes 268-280)

  modules/                 # [À créer Phase 2+]
    admin-users.js         # User management module
    admin-logs.js          # Logs viewer module
    admin-cache.js         # Cache management module
    admin-ml.js            # ML models module
    admin-apikeys.js       # API keys module
```

---

## 🚀 Endpoints API

### Phase 1 (Infrastructure) - ✅ Opérationnel

| Endpoint | Méthode | Description | RBAC |
|----------|---------|-------------|------|
| `/admin/health` | GET | Health check admin | ✅ Admin |
| `/admin/status` | GET | Stats système (users, logs, cache, ML) | ✅ Admin |
| `/admin/users` | GET | Liste tous les utilisateurs | ✅ Admin |
| `/admin/logs/list` | GET | Liste fichiers logs disponibles | ✅ Admin |
| `/admin/cache/stats` | GET | Stats cache (placeholder) | ✅ Admin |
| `/admin/cache/clear` | DELETE | Clear cache par type | ✅ Admin |
| `/admin/ml/models` | GET | Liste modèles ML (placeholder) | ✅ Admin |
| `/admin/apikeys` | GET | Liste clés API (placeholder) | ✅ Admin |

### Phase 2 (User Management + Logs) - 🟡 À venir

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `POST /admin/users` | POST | Créer user + dossier structure |
| `PUT /admin/users/{user_id}` | PUT | Modifier user (label, rôles) |
| `DELETE /admin/users/{user_id}` | DELETE | Supprimer user (soft delete) |
| `POST /admin/users/{user_id}/roles` | POST | Assigner rôles |
| `GET /admin/logs/read` | GET | Lecture logs avec pagination |
| `GET /admin/logs/stats` | GET | Statistiques logs (errors, warnings) |
| `GET /admin/logs/tail` | GET | Tail -f temps réel (SSE) |

### Phase 3 (Cache + ML) - 🔴 À venir

### Phase 4 (API Keys) - 🔴 À venir

---

## 🖥️ Interface Admin Dashboard

### Accès

**URL:** `http://localhost:8080/admin-dashboard.html`

**Prérequis:**
- User avec rôle `admin` sélectionné (ex: "jack")
- Menu "Admin ▾" visible en haut à droite

### Structure

**6 Onglets :**

1. **📊 Overview** - Vue d'ensemble système
   - Stats cards (Total Users, Admin Users, Cache Types, ML Models)
   - Status général

2. **👥 User Management** - ✅ Phase 1 opérationnel
   - Table users avec badges de rôles colorés
   - Colonnes: User ID, Label, Roles, Status, Created
   - [Phase 2] CRUD operations (Create, Edit, Delete)

3. **📝 Logs Viewer** - 🟡 Phase 2
   - Liste fichiers logs
   - Filtres (level, search, date)
   - Pagination
   - Stats (errors, warnings)

4. **⚡ Cache Management** - 🔴 Phase 3
   - Stats par cache type
   - Clear cache
   - Cache warming

5. **🤖 ML Models** - 🔴 Phase 3
   - Liste modèles ML
   - Retraining jobs
   - Versioning

6. **🔑 API Keys** - 🔴 Phase 4
   - Liste clés API (masquées)
   - Usage statistics

### Navigation Hash

- `#overview` - Vue d'ensemble
- `#users` - User Management
- `#logs` - Logs Viewer
- `#cache` - Cache Management
- `#ml` - ML Models
- `#apikeys` - API Keys

### Auto-Reload

L'admin dashboard écoute l'événement `activeUserChanged` pour recharger automatiquement les données quand l'utilisateur change.

```javascript
window.addEventListener('activeUserChanged', (event) => {
  // Reload current tab with new user context
  loadTabContent(currentTab);
});
```

---

## 🧪 Tests

### Test RBAC Backend

```powershell
# User admin (jack) - Doit fonctionner
curl "http://localhost:8080/admin/health" -H "X-User: jack"
# → {"ok": true, "data": {"status": "ok", "admin_user": "jack", ...}}

# User viewer (demo) - Doit échouer (403)
curl "http://localhost:8080/admin/health" -H "X-User: demo"
# → {"detail": "Admin role required for this operation"}
```

### Test Frontend

1. Ouvrir `http://localhost:8080/admin-dashboard.html`
2. Sélectionner user "demo" (viewer)
   - ❌ Message d'erreur : "Access denied. Admin role required."
3. Switch vers user "jack" (admin)
   - ✅ Stats cards se remplissent
   - ✅ Onglet "User Management" affiche la table
   - ✅ Menu "Admin ▾" visible

### Test Navigation

1. Menu "Admin ▾" → "Dashboard"
   - ✅ Redirection vers admin-dashboard.html
2. Menu "Admin ▾" → "User Management"
   - ✅ Redirection vers admin-dashboard.html#users
   - ✅ Onglet "User Management" actif
3. Clic sur onglets
   - ✅ Hash URL mis à jour
   - ✅ Contenu onglet chargé

---

## 🔧 Configuration

### Mode Développement

**Menu Admin visible pour tous (dev uniquement) :**

```javascript
// static/components/nav.js:103-112
const checkAdminRole = () => {
  const isDev = location.hostname === 'localhost' ||
                location.hostname === '127.0.0.1' ||
                location.port === '8080';

  if (isDev) {
    console.debug('🔧 Dev mode detected - Admin role forced');
    return true; // Menu visible pour tous
  }
  // Production: vérifier rôles réels
};
```

**Backend bypass RBAC (dev uniquement) :**

```bash
# .env
DEV_OPEN_API=1  # Bypass RBAC checks (DANGER: dev only!)
```

### Production

**Menu Admin :**
- Visible uniquement pour users avec rôles `admin`, `governance_admin`, ou `ml_admin`
- Stocké dans localStorage : `user_roles` (JSON array)

**Endpoints API :**
- Protection RBAC stricte (pas de bypass)
- Vérification rôle `admin` obligatoire
- Logs audit complets

---

## 📊 Statistiques Phase 1

**Backend :**
- 8 endpoints créés
- 1 dependency RBAC (`require_admin_role`)
- 4 rôles définis
- 6 users configurés (1 admin, 5 viewers)

**Frontend :**
- 1 page admin (admin-dashboard.html)
- 6 onglets (1 opérationnel, 5 placeholders)
- 4 stats cards
- Responsive design (mobile, tablet, desktop, XL)

**Code :**
- 274 lignes backend (admin_router.py)
- 73 lignes RBAC (deps.py::require_admin_role)
- 645 lignes frontend (admin-dashboard.html)
- 13 lignes menu (nav.js modifications)

---

## 🚀 Roadmap

### Phase 2 - User Management + Logs Viewer (À venir)

**User Management :**
- [ ] Service `services/user_management.py`
  - [ ] `create_user(user_id, label, roles)` - Auto-create folder structure
  - [ ] `update_user(user_id, data)` - Modify label, roles, status
  - [ ] `delete_user(user_id)` - Soft delete (rename folder)
  - [ ] `assign_roles(user_id, roles)` - Update roles
- [ ] Endpoints CRUD `/admin/users/*`
- [ ] Frontend modals (Create User, Edit User, Delete User)
- [ ] Form validation (user_id alphanumeric + underscore)

**Logs Viewer :**
- [ ] Service `services/log_reader.py`
  - [ ] Parse log format (regex)
  - [ ] Filter by level, date, search
  - [ ] Pagination support
  - [ ] Stats calculation (count by level, module)
- [ ] Endpoints `/admin/logs/*`
- [ ] Frontend filters UI
- [ ] Real-time tail (SSE ou WebSocket)

### Phase 3 - Cache + ML Models (À venir)

**Cache Management :**
- [ ] Service `services/cache_manager.py` (unified)
- [ ] Stats tous caches (in-memory, CoinGecko, Redis)
- [ ] Clear cache par type
- [ ] Cache warming

**ML Models :**
- [ ] Service `services/ml/training_executor.py`
- [ ] Background training jobs (asyncio)
- [ ] Model deployment (TRAINED → DEPLOYED)
- [ ] Model rollback
- [ ] Real-time progress (WebSocket)

### Phase 4 - API Keys Management (À venir)

- [ ] Service `services/key_masker.py`
- [ ] Lecture secrets.json (all users)
- [ ] Update API keys (masked input)
- [ ] Usage statistics
- [ ] Sensitive key masking (show only last 4 chars)

---

## 🔗 Références

**Code Sources :**
- Backend: [api/admin_router.py](../api/admin_router.py)
- RBAC: [api/deps.py::require_admin_role](../api/deps.py#L161-L235)
- Frontend: [static/admin-dashboard.html](../static/admin-dashboard.html)
- Menu: [static/components/nav.js](../static/components/nav.js#L268-L280)
- Config: [config/users.json](../config/users.json)

**Documentation Connexe :**
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture globale
- [RBAC.md](RBAC.md) - Système RBAC détaillé (à créer)
- [SECURITY.md](SECURITY.md) - Sécurité (safe_loader, path traversal)

**Plan Original :**
- [Plan complet Admin Dashboard](../docs/_archive/session_notes/admin_dashboard_plan_2025_12_19.md) (si archivé)

---

**Dernière mise à jour:** 2025-12-19 - Phase 1 terminée
