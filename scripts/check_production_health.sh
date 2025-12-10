#!/bin/bash
# Script de vérification santé production SmartFolio
# Usage: bash scripts/check_production_health.sh

set -e

API_URL="${API_URL:-http://localhost:8080}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🏥 SmartFolio Production Health Check"
echo "=========================================="
echo ""

# 1. API Health Check
echo "1️⃣  Checking API Health..."
if curl -sf "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is healthy${NC}"
    curl -s "$API_URL/health" | jq '.'
else
    echo -e "${RED}❌ API is DOWN${NC}"
    exit 1
fi
echo ""

# 2. Scheduler Status
echo "2️⃣  Checking Scheduler (Price Updates)..."
SCHEDULER_STATUS=$(curl -sf "$API_URL/api/scheduler/health" | jq -r '.data.enabled')
if [ "$SCHEDULER_STATUS" = "true" ]; then
    echo -e "${GREEN}✅ Scheduler is running${NC}"
    curl -s "$API_URL/api/scheduler/health" | jq '.data.jobs | to_entries[] | {job: .key, last_run: .value.last_run, duration: .value.duration_seconds}'
else
    echo -e "${YELLOW}⚠️  Scheduler is disabled (RUN_SCHEDULER != 1)${NC}"
fi
echo ""

# 3. Redis Status
echo "3️⃣  Checking Redis Cache..."
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is responding${NC}"

    # Count cache keys
    PRICE_KEYS=$(redis-cli KEYS 'price:*' | wc -l)
    ML_KEYS=$(redis-cli KEYS 'ml:*' | wc -l)
    TOTAL_KEYS=$(redis-cli DBSIZE | grep -oE '[0-9]+')

    echo "   - Price cache keys: $PRICE_KEYS"
    echo "   - ML cache keys: $ML_KEYS"
    echo "   - Total keys: $TOTAL_KEYS"

    # Check TTL on a sample price key
    SAMPLE_KEY=$(redis-cli KEYS 'price:*' | head -1)
    if [ -n "$SAMPLE_KEY" ]; then
        TTL=$(redis-cli TTL "$SAMPLE_KEY")
        echo "   - Sample price TTL: ${TTL}s (should be ~180s for 3min cache)"
    fi
else
    echo -e "${RED}❌ Redis is not responding${NC}"
fi
echo ""

# 4. Price Freshness
echo "4️⃣  Checking Price Data Freshness..."
PRICE_RESPONSE=$(curl -sf "$API_URL/balances/current?user_id=demo&source=cointracking")
if [ -n "$PRICE_RESPONSE" ]; then
    echo -e "${GREEN}✅ Price data is available${NC}"

    # Show sample of last updated times
    echo "$PRICE_RESPONSE" | jq -r '.items[:3] | .[] | "   - \(.symbol): \(.price_usd) USD (updated: \(.last_updated // "N/A"))"'

    # Check for stale prices (older than 1 hour)
    STALE_COUNT=$(echo "$PRICE_RESPONSE" | jq '[.items[] | select(.last_updated != null and (now - (.last_updated | fromdateiso8601) > 3600))] | length')
    if [ "$STALE_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Found $STALE_COUNT assets with stale prices (>1h old)${NC}"
    fi
else
    echo -e "${RED}❌ Failed to fetch price data${NC}"
fi
echo ""

# 5. Log File Check
echo "5️⃣  Checking Recent Errors in Logs..."
if [ -f "logs/app.log" ]; then
    ERROR_COUNT=$(grep -c "ERROR" logs/app.log 2>/dev/null || echo "0")
    CRITICAL_COUNT=$(grep -c "CRITICAL" logs/app.log 2>/dev/null || echo "0")

    if [ "$CRITICAL_COUNT" -gt 0 ]; then
        echo -e "${RED}❌ Found $CRITICAL_COUNT CRITICAL errors${NC}"
        echo "   Recent critical errors:"
        grep "CRITICAL" logs/app.log | tail -3
    elif [ "$ERROR_COUNT" -gt 10 ]; then
        echo -e "${YELLOW}⚠️  Found $ERROR_COUNT ERROR entries (last 5 shown)${NC}"
        grep "ERROR" logs/app.log | tail -5
    else
        echo -e "${GREEN}✅ Log file healthy ($ERROR_COUNT errors)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Log file not found at logs/app.log${NC}"
fi
echo ""

# 6. Process Check
echo "6️⃣  Checking Python Process..."
if pgrep -f "uvicorn api.main:app" > /dev/null; then
    echo -e "${GREEN}✅ Uvicorn process is running${NC}"
    ps aux | grep "uvicorn api.main" | grep -v grep | awk '{print "   - PID: "$2" | CPU: "$3"% | MEM: "$4"%"}'
else
    echo -e "${RED}❌ Uvicorn process not found${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "📊 Summary"
echo "=========================================="
echo "Check completed at: $(date)"
echo ""
echo "💡 Tips:"
echo "   - Watch logs: tail -f logs/app.log"
echo "   - Scheduler jobs: curl $API_URL/api/scheduler/health | jq '.data.jobs'"
echo "   - Redis monitor: redis-cli MONITOR"
echo ""
