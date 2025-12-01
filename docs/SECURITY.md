# 🔒 Security Policy - SmartFolio

> **Last Updated:** November 24, 2025
> **Version:** 2.9.x
> **Security Level:** 🟢 Production Ready

---

## 📋 Supported Versions

| Version | Supported          | Security Status |
| ------- | ------------------ | --------------- |
| 2.9.x   | ✅ Yes             | Active support  |
| 2.8.x   | ⚠️ Limited         | Critical fixes only |
| < 2.8   | ❌ No              | Unsupported     |

---

## 🚨 Reporting a Vulnerability

### Do NOT Open Public Issues

**Security vulnerabilities should be reported privately.**

**Contact:** Create a private security advisory on GitHub or contact the maintainers directly.

**Expected Response Time:**
- **Critical vulnerabilities:** 24-48 hours
- **High severity:** 1 week
- **Medium/Low severity:** 2-4 weeks

**What to Include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

---

## 🛡️ Security Measures

### 1. Dependency Security

**Automated Scanning:**
- ✅ Weekly scans with [Safety](https://pyup.io/safety/) (CVE database)
- ✅ All 163+ packages kept up-to-date
- ✅ **0 known CVEs** in production dependencies

**Key Dependencies:**
- `fastapi==0.115.0` - Web framework
- `pydantic==2.9.2` - Data validation
- `torch>=2.0.0` - ML framework
- `httpx>=0.24.0` - Secure HTTP client
- `redis>=5.0.0` - Cache & streaming

**Verification:**
```bash
safety scan --output screen
```

---

### 2. Code Security

**Static Analysis:**
- ✅ Automated [Bandit](https://bandit.readthedocs.io/) scans on every commit
- ✅ Focus on HIGH/MEDIUM severity issues
- ✅ **57 issues** (0 HIGH, 24 MEDIUM acceptable)

**Key Protections:**
- ✅ No hardcoded credentials
- ✅ HTTPS for all external API calls
- ✅ Input validation with Pydantic
- ✅ Path traversal protection for ML models

**Verification:**
```bash
bandit -r api/ services/ -ll --format screen
```

---

### 3. ML Model Security

**Safe Model Loading** (`services/ml/safe_loader.py`):
- ✅ **Path traversal protection** - Models only from `cache/ml_pipeline/`
- ✅ **PyTorch weights_only=True** by default (fallback if needed)
- ✅ **No user-uploaded models** - Local training only
- ✅ **Audit logging** for all model loads

**Protected Files:**
- `services/ml/model_registry.py` - Uses `safe_pickle_load()`
- `services/ml/models/regime_detector.py` - Uses `safe_torch_load()`
- `services/ml/models/correlation_forecaster.py` - Uses `safe_torch_load()`
- `services/ml/models/volatility_predictor.py` - Uses `safe_torch_load()`

**Usage Example:**
```python
from services.ml.safe_loader import safe_pickle_load, safe_torch_load

# ✅ SECURE - Validates path is within SAFE_MODEL_DIR
model = safe_pickle_load("cache/ml_pipeline/models/my_model.pkl")

# ✅ SECURE - Tries weights_only=True first
checkpoint = safe_torch_load("cache/ml_pipeline/models/regime.pth")

# ❌ INSECURE - Direct pickle.load() bypasses validation
with open(path, 'rb') as f:
    model = pickle.load(f)  # Don't do this!
```

---

### 4. Data Protection

**Multi-Tenant Isolation:**
- ✅ **User-scoped file system** (`UserScopedFS`)
- ✅ Path traversal protection
- ✅ Isolated data directories per user: `data/users/{user_id}/`

**File Structure:**
```
data/users/
├── demo/
│   ├── config.json
│   ├── cointracking/data/
│   └── saxobank/data/
└── jack/
    ├── config.json
    ├── cointracking/data/
    └── saxobank/data/
```

**Secret Management:**
- ✅ All secrets in `.env` (never committed to git)
- ✅ Environment variables for production
- ✅ API keys rotation recommended quarterly
- ✅ No credentials in logs or error messages

**Example `.env`:**
```bash
# API Keys (NEVER commit this file)
COINTRACKING_API_KEY=your_key_here
COINGECKO_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

### 5. Authentication & Authorization

**Current Implementation:**
- ✅ Header-based user identification (`X-User`)
- ✅ User-scoped file system access
- ✅ No shared data between users

**API Protection:**
```python
from api.deps import get_active_user

@router.get("/endpoint")
async def endpoint(user: str = Depends(get_active_user)):
    # user is automatically validated and isolated
    pass
```

**Frontend:**
```javascript
const activeUser = localStorage.getItem('activeUser') || 'demo';
const response = await fetch('/api/risk/dashboard', {
    headers: { 'X-User': activeUser }
});
```

---

## 📊 Security Audit Results

### Latest Audit: November 24, 2025

**Tools Used:**
- Safety 3.7.0 (dependency scanner)
- Bandit 1.9.1 (static code analyzer)

**Results:**

| Metric | Status | Details |
|--------|--------|---------|
| Dependencies | ✅ **PASS** | 0 CVEs (163 packages) |
| Code HIGH Issues | ✅ **0** | All MD5 usage documented |
| Code MEDIUM Issues | 🟡 **24** | Acceptable (ML context) |
| Code LOW Issues | ℹ️ **33** | Informational only |
| **Overall** | 🟢 **READY** | Production approved |

**Improvements Since Last Audit:**
- ✅ Eliminated 6 HIGH issues (-100%)
- ✅ Reduced 5 MEDIUM issues (-17%)
- ✅ Created safe_loader.py system
- ✅ Migrated urllib → httpx
- ✅ Added usedforsecurity=False to all MD5

**Full Report:** See `SECURITY_AUDIT_2025-11-22.md`

---

## 🔄 Continuous Security

### Automated Scans

**Weekly (Recommended):**
```bash
# Activate virtual environment
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate     # Linux/Mac

# Scan dependencies
safety scan --output screen

# Scan code
bandit -r api/ services/ -ll --format screen
```

**Monthly Tasks:**
- Review and update dependencies
- Check for new CVEs in dependency database
- Audit user access logs
- Review API key usage

**Quarterly Tasks:**
- Rotate all API keys
- Full security audit
- Review and update this security policy

---

## 📚 Security Resources

### Internal Documentation
- [`SECURITY_AUDIT_2025-11-22.md`](../SECURITY_AUDIT_2025-11-22.md) - Latest audit report
- [`services/ml/safe_loader.py`](../services/ml/safe_loader.py) - Safe ML model loading
- [`CLAUDE.md`](../CLAUDE.md) - Development guidelines

### External References
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Safety Documentation](https://docs.safetycli.com/)
- [PyTorch Security](https://pytorch.org/docs/stable/notes/serialization.html#security)
- [Python Pickle Security](https://docs.python.org/3/library/pickle.html#module-pickle)
- [HTTPS Best Practices](https://developer.mozilla.org/en-US/docs/Web/Security)

---

## ✅ Security Checklist

### Before Each Release

- [ ] Run `safety scan` - No new CVEs
- [ ] Run `bandit -r api/ services/ -ll` - No new HIGH issues
- [ ] Review all `.env` variables - No secrets in code
- [ ] Check git history - No committed credentials
- [ ] Test multi-tenant isolation - Users can't access each other's data
- [ ] Verify HTTPS endpoints - All external APIs use HTTPS
- [ ] Review logs - No sensitive data logged
- [ ] Update dependencies - All packages up-to-date

### Before Production Deployment

- [ ] Full security audit completed
- [ ] All HIGH severity issues resolved
- [ ] Documentation updated
- [ ] Backup strategy in place
- [ ] Monitoring & alerting configured
- [ ] Incident response plan documented

---

## 🚀 Incident Response

### In Case of Security Incident

1. **Immediate Actions:**
   - Assess severity and impact
   - Contain the incident (isolate affected systems)
   - Document all findings

2. **Investigation:**
   - Identify root cause
   - Determine affected users/data
   - Review logs and audit trails

3. **Remediation:**
   - Apply security patches
   - Rotate compromised credentials
   - Update security measures

4. **Communication:**
   - Notify affected users (if applicable)
   - Report to security team
   - Update incident log

5. **Post-Incident:**
   - Conduct post-mortem analysis
   - Update security policy
   - Implement preventive measures

---

## 🧑‍💻 Developer Security Guide

This section describes the security measures implemented in SmartFolio and the best practices to follow.

### 1. Protection des credentials

- ❌ **Jamais committer** `.env` avec des vraies clés API
- ✅ Utiliser `.env.example` comme template sanitisé
- ✅ Pre-commit hooks avec `gitleaks` et `detect-secrets`
- ✅ `.gitignore` configuré pour bloquer `.env`

### 2. Headers de sécurité HTTP

Notre API expose automatiquement ces headers de sécurité :

```http
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; ...
X-Content-Type-Options: nosniff
X-Frame-Options: SAMEORIGIN
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
```

### 3. Code Quality & Logs

- ✅ ESLint configuré avec `no-console` et `no-eval`
- ✅ Migration automatique `console.log` → `debugLogger`
- ✅ Logs conditionnels (silencieux en production)
- ✅ Tests automatisés pour headers de sécurité

### 4. API Security

- ✅ Rate limiting configuré
- ✅ CORS restrictif (pas de `*` en production)
- ✅ Validation Pydantic sur tous les endpoints
- ✅ Gestion d'erreurs sans fuite d'informations

## 🛠 Outils de sécurité

### Scan automatique des secrets

```bash
# Installation
pip install detect-secrets pre-commit
pre-commit install

# Scan manuel
detect-secrets scan --baseline .secrets.baseline
gitleaks detect --verbose
```

### Audit de sécurité

```powershell
# Script d'audit complet
.\tools\security-audit.ps1
```

### Tests de sécurité

```bash
# Tests des headers HTTP
pytest tests/test_security_headers.py -v

# Smoke tests étendus
python tests/smoke_test_refactored_endpoints.py
```

## 🚀 Configuration de production

### Variables d environnement critiques

```bash
# .env (NE JAMAIS COMMITTER)
DEBUG=false
COINGECKO_API_KEY=your_real_key_here
FRED_API_KEY=your_real_key_here
CT_API_KEY=your_real_key_here
CT_API_SECRET=your_real_secret_here
DEBUG_TOKEN=strong_random_token_for_debug_endpoints
```

### Headers CSP strictes

En production, s assurer que la CSP ne contient pas `unsafe-inline` ou `unsafe-eval` sans nonce approprié.

### Rate limiting

```python
# config/settings.py
RATE_LIMIT_PER_MINUTE = 100  # Ajuster selon le trafic
```

## 🔍 Monitoring de sécurité

### Endpoints de santé sécurisés

- `GET /api/ml/status` - Statut ML (sans infos sensibles)
- `GET /api/risk/status` - Statut risk management
- `GET /api/alerts/active` - Alertes actives (authentifiées)

### Endpoints d admin protégés

- `GET /api/ml/debug/*` - Nécessite `X-Admin-Key`
- `POST /api/execution/approve/*` - Nécessite authentification

## ⚠️ Pratiques à éviter

### ❌ Ne pas faire

```javascript
// MAUVAIS - Log sensible
console.log('API Key:', apiKey);

// MAUVAIS - Eval dynamique
eval(userInput);

// MAUVAIS - Headers permissifs
"Access-Control-Allow-Origin": "*"
```

### ✅ Faire plutôt

```javascript
// BON - Log conditionnel
debugLogger.info('API call successful');

// BON - Validation stricte
const validated = UserInputSchema.parse(input);

// BON - CORS restrictif
"Access-Control-Allow-Origin": "https://mondomaine.com"
```

## 📋 Checklist de sécurité

Avant chaque déploiement :

- [ ] `.env` non commité
- [ ] Clés API révoquées/régénérées si exposées
- [ ] Pre-commit hooks activés
- [ ] `.\tools\security-audit.ps1` passe
- [ ] Tests de sécurité verts
- [ ] CSP configurée sans `unsafe-*`
- [ ] Rate limiting activé
- [ ] Logs de debug désactivés en production

## 🚨 En cas d incident

### Clés API compromises

1. **Immédiatement** révoquer les clés dans les services externes
2. Générer de nouvelles clés
3. Purger l historique git si nécessaire :
   ```bash
   git filter-repo --invert-paths --path fichier_avec_secrets.py
   ```
4. Notifier l équipe

### Vulnérabilité détectée

1. Évaluer la criticité
2. Appliquer un correctif temporaire si nécessaire
3. Développer et tester le correctif permanent
4. Déployer et vérifier

## 📚 Ressources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [CSP Generator](https://report-uri.com/home/generate)
- [Git Secrets Detection](https://github.com/awslabs/git-secrets)

---

**⚡ Règle d or :** En cas de doute sur la sécurité, toujours choisir l option la plus restrictive.