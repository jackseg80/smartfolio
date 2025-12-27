# 🚀 Reprendre Ici - Session AI Chat (Dec 27, 2025)

> **TL;DR:** AI Chat fonctionne à 90%. Context builders dashboard/risk à améliorer.

---

## ⚡ Quick Start (5 min)

### 1. Lire Document Handoff
📖 **[docs/AI_CHAT_HANDOFF_DEC_27.md](docs/AI_CHAT_HANDOFF_DEC_27.md)** (document complet)

### 2. Problèmes à Résoudre

**Dashboard Context:**
- ❌ Ne voit QUE crypto (pas bourse, patrimoine, risk score, régimes)
- 🔧 Fichier: `static/components/ai-chat-context-builders.js` lignes 9-66
- ✅ Solution: Appels API directs (détails dans handoff)

**Risk Dashboard Context:**
- ❌ IA dit "pas accès aux données" alors que API fonctionne
- 🔧 Fichier: `static/components/ai-chat-context-builders.js` lignes 71-130
- ✅ Solution: Debug + logs (détails dans handoff)

---

## 📝 Checklist 1h

- [ ] **Debug (15 min)**
  - Tester `window.getUnifiedState()` (Console F12)
  - Identifier endpoints API (curl)

- [ ] **Code (30 min)**
  - Enrichir `buildDashboardContext()`
  - Debug `buildRiskDashboardContext()`

- [ ] **Test (15 min)**
  - Quick Test avec user `jack`
  - Vérifier Console F12

- [ ] **Commit (10 min)**
  - Push fixes
  - Merger PR

---

## 🔗 Références

- **Handoff complet:** [docs/AI_CHAT_HANDOFF_DEC_27.md](docs/AI_CHAT_HANDOFF_DEC_27.md)
- **Quick Test:** [docs/AI_CHAT_QUICK_TEST.md](docs/AI_CHAT_QUICK_TEST.md)
- **Context Builders:** [static/components/ai-chat-context-builders.js](static/components/ai-chat-context-builders.js)

---

**Priorité:** Dashboard context (cross-asset)
**Temps:** ~1h
**User test:** `jack` (pas demo)
