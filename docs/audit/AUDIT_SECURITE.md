# Audit Sécurité - SmartFolio

**Date:** 9 novembre 2025
**Scope:** OWASP Top 10 + Vulnérabilités générales
**Note Globale:** 6/10 - Moyen avec vulnérabilités critiques

---

## RÉSUMÉ

- **Vulnérabilités Critiques:** 3
- **Vulnérabilités Hautes:** 5
- **Vulnérabilités Moyennes:** 8
- **Vulnérabilités Basses:** 4
- **Total:** 20 issues identifiées

---

## VULNÉRABILITÉS CRITIQUES

### 1. Clé API Exposée dans .env

**Sévérité:** CRITIQUE
**OWASP:** A02:2021 - Cryptographic Failures
**Fichier:** `.env:10`

**Problème:**
```bash
COINGECKO_API_KEY=CG-ZcsKJgLUH5DeU2xeSu7R2a6v
```

**Exploitation:**
- Attaquant accède au filesystem → vole clé → appels API illimités
- Quota drainé, coûts supplémentaires

**Remédiation:**
1. **IMMÉDIATEMENT** révoquer `CG-ZcsKJgLUH5DeU2xeSu7R2a6v`
2. Générer nouvelle clé
3. Migrer vers secret manager (Azure Key Vault / AWS Secrets Manager)
4. Vérifier historique git: `git log --all --full-history -- .env`

---

### 2. Credentials Hardcodés

**Sévérité:** CRITIQUE
**OWASP:** A07:2021 - Identification and Authentication Failures
**Fichiers:** 3 fichiers

**Occurrences:**
```python
# api/unified_ml_endpoints.py:486
expected_key = os.getenv("ADMIN_KEY", "crypto-rebal-admin-2024")

# tests/smoke_test_refactored_endpoints.py:147
headers = {"X-Admin-Key": "crypto-rebal-admin-2024"}

# setup_dev.py:122
DEBUG_TOKEN=dev-secret-2024
```

**Exploitation:**
- Attaquant lit code public → trouve credentials → accès admin
- Endpoints `/api/ml/debug/*` compromis

**Remédiation:**
```python
# AVANT
expected_key = os.getenv("ADMIN_KEY", "crypto-rebal-admin-2024")

# APRÈS
expected_key = os.getenv("ADMIN_KEY")
if not expected_key:
    raise ValueError("ADMIN_KEY environment variable required")

# Générer token fort:
# openssl rand -hex 32
```

---

### 3. eval() en JavaScript

**Sévérité:** CRITIQUE
**OWASP:** A03:2021 - Injection
**Fichier:** `static/modules/risk-dashboard-main-controller.js:3724`

**Code vulnérable:**
```javascript
const onclickAttr = event.target.getAttribute('onclick');
if (onclickAttr) {
  try {
    eval(onclickAttr);  // DANGER!
  } catch (error) {
    debugLogger.error('Error executing toast action:', error);
  }
}
```

**Exploitation:**
1. Attaquant injecte `onclick="maliciousCode()"`
2. Utilisateur clique → code arbitraire exécuté
3. Vol localStorage, redirection phishing, actions admin

**Remédiation:**
```javascript
// [OK] Solution sécurisée: Event delegation
const TOAST_ACTIONS = {
  'reload': () => location.reload(),
  'dismiss': () => dismissToast(),
  'viewDetails': () => showDetails()
};

const actionName = event.target.getAttribute('data-action');
if (TOAST_ACTIONS[actionName]) {
  TOAST_ACTIONS[actionName]();
}
```

---

## VULNÉRABILITÉS HAUTES

### 4. CORS Wildcard en Dev

**Sévérité:** HAUTE
**Fichiers:** `start_simple.py:18`, `tests/unit/test_risk_server.py:25`

**Problème:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accepte TOUT domaine
    allow_credentials=True,  # Inclut cookies!
    allow_headers=["*"]
)
```

**Exploitation:**
- Attaquant héberge site malveillant `evil.com`
- Victime visite `evil.com` (logged in SmartFolio)
- JavaScript fait appels API avec credentials victime
- Vol données portfolio, exécution trades

**Remédiation:**
```python
# Même en dev, restreindre
allow_origins=[
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]
```

---

### 5. Bypass Auth en DEV Mode

**Sévérité:** HAUTE
**Fichier:** `api/deps.py:49-52`

**Code:**
```python
dev_mode = os.getenv("DEV_OPEN_API", "0") == "1"
if dev_mode:
    logger.info(f"DEV MODE: Bypassing authorization for user: {normalized_user}")
    return normalized_user  # AUCUNE VÉRIFICATION!
```

**Exploitation:**
- Dev oublie de désactiver `DEV_OPEN_API=1` en staging/prod
- Attaquant envoie `X-User: admin` → accès à TOUS comptes
- Bypass complet isolation multi-tenant

**Remédiation:**
```python
# Validation au démarrage
if os.getenv("DEV_OPEN_API") == "1" and os.getenv("ENVIRONMENT") == "production":
    raise RuntimeError("DEV_OPEN_API cannot be enabled in production!")

# Log warning visible
if dev_mode:
    logger.warning(" DEV MODE ACTIVE - AUTHENTICATION BYPASSED")
```

---

### 6. Pickle Insecure Deserialization

**Sévérité:** HAUTE
**Fichiers:** 9 occurrences

**Locations:**
- `services/ml_pipeline_manager_optimized.py` (7 occurrences)
- `services/ml_models.py:394`
- `scripts/train_models.py` (3 occurrences)

**Problème:**
```python
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)  # UNSAFE!
```

**Exploitation:**
- Attaquant upload `.pkl` malveillant
- `pickle.load()` → exécution code arbitraire (RCE)
- Backdoor, exfiltration données

**Remédiation:**
```python
# Option 1: Validation hash
import hashlib

def load_model_safe(path, expected_hash):
    with open(path, 'rb') as f:
        content = f.read()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("Model tampering detected")
        return pickle.loads(content)

# Option 2: Format plus sûr (ONNX, TorchScript)
```

---

### 7. Command Injection (shell=True)

**Sévérité:** HAUTE
**Fichiers:** `deploy.py:103`, `test_phase1_simple.py:23`

**Code:**
```python
result = subprocess.run(command, shell=True, capture_output=True)
```

**Exploitation:**
- Si `command` contient input utilisateur
- Injection: `filename.csv; rm -rf / #`
- Exécution commandes arbitraires

**Remédiation:**
```python
# AVANT
subprocess.run(f"python {script}", shell=True)

# APRÈS
subprocess.run(["python", script], shell=False)

# Si shell requis, sanitiser:
import shlex
safe_script = shlex.quote(script)
```

---

### 8. File Upload Insuffisamment Validé

**Sévérité:** HAUTE
**Fichier:** `api/sources_endpoints.py:518-524`

**Problème:**
```python
safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
with open(file_path, 'wb') as f:
    f.write(content)
```

**Issues:**
1. Pas de validation MIME type
2. Pas de scan malware
3. Extension vérifiée mais pas contenu
4. Pas de rate limiting uploads

**Remédiation:**
```python
# Valider MIME type
import magic
mime = magic.from_buffer(content, mime=True)
if mime != 'text/csv':
    raise ValueError("Invalid file type")

# Valider contenu CSV
import csv
try:
    csv.reader(io.StringIO(content.decode('utf-8')))
except:
    raise ValueError("Invalid CSV format")

# Ajouter ClamAV scan en production
```

---

## VULNÉRABILITÉS MOYENNES

### 9. Pas de Protection CSRF

**Sévérité:** MOYENNE
**Status:** Non implémenté

**Problème:**
FastAPI n'a pas de protection CSRF native pour POST/PUT/DELETE.

**Exploitation:**
```html
<!-- Site attaquant -->
<form action="http://localhost:8080/api/sources/upload" method="POST">
  <input type="file" name="file" value="malicious.csv">
</form>
<script>document.forms[0].submit()</script>
```

**Remédiation:**
```python
# Implémenter CSRF tokens
from fastapi_csrf_protect import CsrfProtect

@app.post("/upload")
async def upload(csrf_protect: CsrfProtect = Depends()):
    await csrf_protect.validate_csrf()
```

---

### 10. innerHTML (XSS Potentiel)

**Sévérité:** MOYENNE
**Occurrences:** 28 fichiers JavaScript

**Exemple:**
```javascript
card.innerHTML = `<div>${userData}</div>`; // Si userData = user input → XSS
```

**Remédiation:**
```javascript
// [OK] Option 1: textContent
card.textContent = userData;  // Auto-escape

// [OK] Option 2: DOMPurify
import DOMPurify from 'dompurify';
card.innerHTML = DOMPurify.sanitize(userData);
```

---

### 11-16. Autres Vulnérabilités Moyennes

11. **Pas de redirection HTTPS en dev** - Risque interception credentials
12. **Logging données sensibles** - Premiers 8 chars API keys visibles
13. **Wildcard allowed_hosts en dev** - Host header injection
14. **Pas de rate limiting auth** - Brute force possible
15. **Path traversal (mitigé)** - Bien protégé mais à tester
16. **DEBUG=true dans .env** - Ne devrait pas être committée

---

## VULNÉRABILITÉS BASSES

17. **Debug mode en .env** - Devrait être .env.example
18. **Redis sans auth** - Password recommandé
19. **Info disclosure errors** - Stack traces en debug
20. **Security headers manquants** - HSTS, Permissions-Policy

---

## POINTS POSITIFS

1. [OK] Protection path traversal excellente (`user_fs.py`)
2. [OK] `.gitignore` bien configuré
3. [OK] Multi-tenant isolation solide
4. [OK] Input sanitization filenames
5. [OK] Dependency injection auth
6. [OK] Rate limiting implémenté
7. [OK] CORS restreint dans main app

---

## PLAN D'ACTION

### Semaine 1 (CRITIQUE)
- [ ] Révoquer clé CoinGecko
- [ ] Supprimer credentials hardcodés
- [ ] Remplacer eval() JavaScript
- [ ] Fix CORS wildcard
- [ ] Validation DEV_OPEN_API production

### Semaines 2-3 (HAUTE)
- [ ] Implémenter CSRF protection
- [ ] Valider hash modèles pickle
- [ ] Supprimer shell=True
- [ ] Améliorer validation uploads
- [ ] JWT pour WebSocket

### Mois 2 (MOYENNE)
- [ ] Auditer tous innerHTML
- [ ] HTTPS en dev
- [ ] Rate limiting auth endpoints
- [ ] Sanitiser logs sensibles
- [ ] Ajouter security headers manquants

---

## MÉTRIQUES

**Avant corrections:**
- [Negative] 3 Critiques
- [Pending] 5 Hautes
- [Pending] 8 Moyennes
- [Positive] 4 Basses
- **Score:** 6/10

**Après Semaine 1:**
- [Negative] 0 Critiques [OK]
- [Pending] 5 Hautes
- **Score:** 7.5/10

**Après 1 Mois:**
- [Negative] 0 Critiques [OK]
- [Pending] 0 Hautes [OK]
- [Pending] 3-4 Moyennes
- **Score:** 9/10 (Production Ready)

---

## RÉFÉRENCES

- [OWASP Top 10 2021](https://owasp.org/Top10/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- Documentation projet: `docs/SECURITY.md`

---

**Rapport généré par:** Claude Code Agent - Security Analysis
