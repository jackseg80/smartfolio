# Wealth Module - Documentation

> **Version:** 2.0 (Feb 2026)
> **Status:** Production Ready
> **Previous name:** Patrimoine Module (renamed in Feb 2026 FR→EN migration)

## Overview

The **Wealth** module unifies the management of all personal assets and liabilities beyond cryptocurrencies and stocks. It replaces and extends the former "Banks" module with a flexible architecture for tracking:

- **Liquidity**: Bank accounts, neobanks, cash
- **Tangible Assets**: Real estate, vehicles, precious metals
- **Liabilities**: Mortgages, personal loans, credit cards
- **Insurance**: Life insurance, investment insurance

## Architecture

### Backend

**Pydantic Models** ([models/wealth.py](../models/wealth.py:121-162))
```python
class WealthItemInput(BaseModel):
    name: str                    # e.g., "House Lyon 120m²"
    category: Literal[...]       # liquidity | tangible | liability | insurance
    type: Literal[...]          # bank_account | real_estate | mortgage | ...
    value: float                 # Value (positive for assets, negative for liabilities)
    currency: str                # CHF | EUR | USD | GBP
    acquisition_date: Optional[str]
    notes: Optional[str]
    metadata: Optional[dict]     # Type-specific fields (flexible JSON)

class WealthItemOutput(WealthBaseModel):
    # All fields from Input +
    id: str                      # Generated UUID
    value_usd: Optional[float]   # Automatic USD conversion
```

**Backward compatibility aliases:**
```python
PatrimoineItemInput = WealthItemInput
PatrimoineItemOutput = WealthItemOutput
```

**CRUD Service** ([services/wealth/wealth_service.py](../services/wealth/wealth_service.py))
- `list_items(user_id, category=None, type=None)` - List with filters
- `get_item(user_id, item_id)` - Get by ID
- `create_item(user_id, item)` - Create
- `update_item(user_id, item_id, item)` - Update
- `delete_item(user_id, item_id)` - Delete
- `get_summary(user_id)` - Net Worth + breakdown by category

**Migration Service** ([services/wealth/wealth_migration.py](../services/wealth/wealth_migration.py))
- Automatic migration `banks/snapshot.json` → `wealth/wealth.json`
- Preserves the original file (read-only)
- Transforms: `bank_account` → `category=liquidity, type=bank_account`

**API Endpoints** ([api/wealth_endpoints.py](../api/wealth_endpoints.py))
```
GET    /api/wealth/items          # List items (category/type filters)
GET    /api/wealth/items/{id}     # Get item
POST   /api/wealth/items          # Create
PUT    /api/wealth/items/{id}     # Update
DELETE /api/wealth/items/{id}     # Delete
GET    /api/wealth/summary        # Net Worth summary
```

**Legacy routes** (still supported for backward compatibility):
```
GET    /api/wealth/patrimoine/items          # Redirects to /api/wealth/items
POST   /api/wealth/patrimoine/items          # Redirects to /api/wealth/items
GET    /api/wealth/patrimoine/summary        # Redirects to /api/wealth/summary
```

**Bank account backward compatibility** ([api/wealth_endpoints.py](../api/wealth_endpoints.py))
```
GET    /api/wealth/banks/accounts    # Redirects to wealth (filter bank_account)
POST   /api/wealth/banks/accounts    # Converts BankAccountInput → WealthItem
PUT    /api/wealth/banks/accounts/{id}
DELETE /api/wealth/banks/accounts/{id}
```

### Frontend

**Dashboard** ([static/wealth-dashboard.html](../static/wealth-dashboard.html))

Structure:
```
┌─────────────────────────────────────┐
│  Net Worth Dashboard                │
│  • Net Worth: $XXX,XXX              │
│  • Total Assets / Liabilities       │
│  • Breakdown by category            │
└─────────────────────────────────────┘

┌──────────┬───────────┬──────────┬──────────┐
│Liquidity │ Tangible  │Liabilities│Insurance│ ← Tabs
└──────────┴───────────┴──────────┴──────────┘
```

**UI Features:**
- Responsive tab system
- Tables by asset type (bank accounts, real estate, mortgages, etc.)
- CRUD modal with dynamic fields per type
- Automatic USD conversion
- Multi-user isolation (localStorage activeUser)
- Full-width responsive design

**Navigation** ([static/components/nav.js](../static/components/nav.js))
- Main menu: "Wealth" link → `wealth-dashboard.html`

**Redirects** ([static/banks-manager.html](../static/banks-manager.html), [static/banks-dashboard.html](../static/banks-dashboard.html))
- Legacy files converted to automatic redirects
- Redirect to `wealth-dashboard.html?tab=liquidity`

**Main Dashboard** ([static/dashboard.html](../static/dashboard.html))
- "Wealth" tile in the main dashboard (Level 1)
- Shows: Net Worth, Assets/Liabilities, Items count
- Doughnut chart with breakdown by category
- Export Lists button

**Global Overview** ([api/wealth_endpoints.py](../api/wealth_endpoints.py))
- Integration in `/api/wealth/global/summary`
- Shows **Net Worth** (assets - liabilities) in the global breakdown
- Consistent with Crypto and Stocks modules

## Data

### Multi-Tenant Storage

**File structure:**
```
data/users/{user_id}/
  banks/
    snapshot.json           # LEGACY (read-only, auto migration)
  wealth/
    wealth.json             # PRIMARY (unified format)
    patrimoine.json         # LEGACY fallback (if wealth.json not found)
```

**Format wealth.json:**
```json
{
  "items": [
    {
      "id": "uuid",
      "name": "House Lyon 120m²",
      "category": "tangible",
      "type": "real_estate",
      "value": 450000.0,
      "currency": "EUR",
      "acquisition_date": "2020-01-15",
      "notes": "Primary residence",
      "metadata": {
        "address": "123 rue de Lyon",
        "surface": 120,
        "monthly_payment": 1200,
        "remaining_loan": 200000
      }
    }
  ]
}
```

### Categories & Types

**4 main categories:**

1. **liquidity** (Liquid Assets)
   - `bank_account` - Bank account
   - `neobank` - Neobank (Revolut, N26, etc.)
   - `cash` - Cash

2. **tangible** (Tangible Assets)
   - `real_estate` - Real estate
   - `vehicle` - Vehicle
   - `precious_metals` - Precious metals

3. **liability** (Liabilities - negative values)
   - `mortgage` - Mortgage
   - `loan` - Personal loan
   - `credit_card` - Credit card

4. **insurance** (Insurance)
   - `life_insurance` - Life insurance
   - `investment_insurance` - Investment insurance

### Type-Specific Metadata

**bank_account:**
```json
{
  "bank_name": "UBS",
  "account_type": "current|savings|pel|livret_a|other"
}
```

**real_estate:**
```json
{
  "address": "123 rue de Lyon",
  "surface": 120,
  "monthly_payment": 1200,
  "remaining_loan": 200000
}
```

**vehicle:**
```json
{
  "brand": "Tesla",
  "model": "Model 3",
  "year": 2023
}
```

**mortgage / loan:**
```json
{
  "monthly_payment": 1200,
  "remaining_months": 180
}
```

## Usage

### Backend

**Create an item:**
```python
from services.wealth.wealth_service import create_item
from models.wealth import WealthItemInput

item = WealthItemInput(
    name="House Lyon",
    category="tangible",
    type="real_estate",
    value=450000.0,
    currency="EUR",
    metadata={"surface": 120, "address": "Lyon"}
)

created = create_item(user_id="jack", item=item)
print(f"Created: {created.id} - ${created.value_usd}")
```

**Get Net Worth:**
```python
from services.wealth.wealth_service import get_summary

summary = get_summary(user_id="jack")
print(f"Net Worth: ${summary['net_worth']:.2f}")
print(f"Assets: ${summary['total_assets']:.2f}")
print(f"Liabilities: ${summary['total_liabilities']:.2f}")
```

### Frontend

**Load data:**
```javascript
const activeUser = localStorage.getItem('activeUser') || 'demo';

// Summary
const summaryRes = await fetch('/api/wealth/summary', {
    headers: { 'X-User': activeUser }
});
const summary = await summaryRes.json();

// Items
const itemsRes = await fetch('/api/wealth/items', {
    headers: { 'X-User': activeUser }
});
const items = await itemsRes.json();
```

**Create an item:**
```javascript
const payload = {
    name: "Revolut",
    category: "liquidity",
    type: "neobank",
    value: 5000.0,
    currency: "EUR",
    notes: "Neobank travel"
};

const response = await fetch('/api/wealth/items', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-User': activeUser
    },
    body: JSON.stringify(payload)
});
```

## Migration

### Automatic Migration

Migration triggers automatically on first access for each user.

**Manual command (Python):**
```python
from services.wealth.wealth_migration import migrate_user_data, migrate_all_users

# Single user
result = migrate_user_data("jack", force=False)

# All users
results = migrate_all_users(force=False)
```

**Transformation:**
```
BEFORE (banks/snapshot.json):
{
  "accounts": [
    {
      "id": "uuid",
      "bank_name": "UBS",
      "account_type": "current",
      "balance": 5000.0,
      "currency": "CHF"
    }
  ]
}

AFTER (wealth/wealth.json):
{
  "items": [
    {
      "id": "uuid",
      "name": "UBS (current)",
      "category": "liquidity",
      "type": "bank_account",
      "value": 5000.0,
      "currency": "CHF",
      "metadata": {
        "bank_name": "UBS",
        "account_type": "current"
      }
    }
  ]
}
```

## Tests

**Unit tests** ([tests/](../tests/))

```bash
# Run all tests
pytest -q tests/unit && pytest -q tests/integration
```

**Manual API tests:**
```bash
# Summary
curl -H "X-User: jack" http://localhost:8080/api/wealth/summary

# List items
curl -H "X-User: jack" http://localhost:8080/api/wealth/items

# Create real estate item
curl -X POST http://localhost:8080/api/wealth/items \
  -H "X-User: jack" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "House Lyon",
    "category": "tangible",
    "type": "real_estate",
    "value": 450000.0,
    "currency": "EUR",
    "metadata": {"surface": 120}
  }'
```

## Future Enhancements

**Phase 3 - Analytics (optional):**
- [ ] Net Worth evolution graph (30d, 1y history)
- [ ] Pie chart asset breakdown by category
- [ ] Trend analysis (growing/declining wealth)

**Phase 4 - Advanced Features:**
- [ ] Photo/attachment support for real estate
- [ ] Value history tracking (track modifications)
- [ ] CSV/PDF/Excel export
- [ ] Loan maturity alerts
- [ ] Real estate ROI calculator

## References

**Code:**
- Backend: [services/wealth/](../services/wealth/)
- Frontend: [static/wealth-dashboard.html](../static/wealth-dashboard.html)
- API: [api/wealth_endpoints.py](../api/wealth_endpoints.py)
- Models: [models/wealth.py](../models/wealth.py)

**Related Docs:**
- [CLAUDE.md](../CLAUDE.md) - Main agent guide
- [architecture.md](architecture.md) - Global architecture

---

**Contributors:** Claude Code (Nov 2025), Updated Feb 2026 (FR→EN migration)
**Maintenance:** Production-ready module, no critical maintenance required
