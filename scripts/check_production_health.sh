#!/bin/bash
# Script de vérification santé production SmartFolio
# Usage: bash scripts/check_production_health.sh

set -e

# Detect server IP automatically (production uses LAN IP, not localhost)
SERVER_IP=$(hostname -I | awk '{print $1}')
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="localhost"
fi

API_URL="${API_URL:-http://${SERVER_IP}:8080}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo " SmartFolio Production Health Check"
echo "=========================================="
echo "API URL: $API_URL"
echo ""

# 1. API Health Check
echo "1.  Checking API Health..."
if curl -sf "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}[OK] API is healthy${NC}"
    curl -s "$API_URL/health" | jq '.'
else
    echo -e "${RED}[Error] API is DOWN${NC}"
    exit 1
fi
echo ""

# 2. Scheduler Status
echo "2.  Checking Scheduler (Price Updates)..."
SCHEDULER_STATUS=$(curl -sf "$API_URL/api/scheduler/health" | jq -r '.data.enabled')
if [ "$SCHEDULER_STATUS" = "true" ]; then
    echo -e "${GREEN}[OK] Scheduler is running${NC}"
    curl -s "$API_URL/api/scheduler/health" | jq '.data.jobs | to_entries[] | {job: .key, last_run: .value.last_run, duration: .value.duration_seconds}'
else
    echo -e "${YELLOW}[Warning]  Scheduler is disabled (RUN_SCHEDULER != 1)${NC}"
fi
echo ""

# 3. Redis Status
echo "3.  Checking Redis Cache..."
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}[OK] Redis is responding${NC}"

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
    echo -e "${RED}[Error] Redis is not responding${NC}"
fi
echo ""

# 4. Price Freshness (using wealth API endpoint)
echo "4.  Checking Price Data Freshness..."
PRICE_RESPONSE=$(curl -sf "$API_URL/api/wealth/global/summary?source=cointracking_api" -H "X-User: jack")
if [ -n "$PRICE_RESPONSE" ]; then
    TOTAL_VALUE=$(echo "$PRICE_RESPONSE" | jq -r '.total_value_usd // 0')

    if [ "$TOTAL_VALUE" != "0" ] && [ "$TOTAL_VALUE" != "null" ]; then
        echo -e "${GREEN}[OK] Price data is available${NC}"

        # Show wealth breakdown
        CRYPTO_TOTAL=$(echo "$PRICE_RESPONSE" | jq -r '.breakdown.crypto // 0')
        SAXO_TOTAL=$(echo "$PRICE_RESPONSE" | jq -r '.breakdown.saxo // 0')
        PATRIMOINE_TOTAL=$(echo "$PRICE_RESPONSE" | jq -r '.breakdown.patrimoine // 0')
        PNL_TODAY=$(echo "$PRICE_RESPONSE" | jq -r '.pnl_today // 0')
        PNL_PCT=$(echo "$PRICE_RESPONSE" | jq -r '.pnl_today_pct // 0')

        echo "   - Crypto: \$$(printf '%.2f' $CRYPTO_TOTAL) USD"
        echo "   - Saxo: \$$(printf '%.2f' $SAXO_TOTAL) USD"
        echo "   - Patrimoine: \$$(printf '%.2f' $PATRIMOINE_TOTAL) USD"
        echo "   - Grand Total: \$$(printf '%.2f' $TOTAL_VALUE) USD"
        echo "   - P&L Today: \$$(printf '%.2f' $PNL_TODAY) ($(printf '%.2f' $PNL_PCT)%)"
    else
        echo -e "${YELLOW}[Warning]  API returned zero or null value${NC}"
        echo "$PRICE_RESPONSE" | jq '.'
    fi
else
    echo -e "${RED} Failed to fetch price data${NC}"
fi
echo ""

# 5. Log File Check
echo "5.  Checking Recent Errors in Logs..."
if [ -f "logs/app.log" ]; then
    ERROR_COUNT=$(grep "ERROR" logs/app.log 2>/dev/null | wc -l)
    CRITICAL_COUNT=$(grep "CRITICAL" logs/app.log 2>/dev/null | wc -l)

    if [ "$CRITICAL_COUNT" -gt 0 ]; then
        echo -e "${RED}[Error] Found $CRITICAL_COUNT CRITICAL errors${NC}"
        echo "   Recent critical errors:"
        grep "CRITICAL" logs/app.log | tail -3
    elif [ "$ERROR_COUNT" -gt 10 ]; then
        echo -e "${YELLOW}[Warning]  Found $ERROR_COUNT ERROR entries (last 5 shown)${NC}"
        grep "ERROR" logs/app.log | tail -5
    else
        echo -e "${GREEN}[OK] Log file healthy ($ERROR_COUNT errors)${NC}"
    fi
else
    echo -e "${YELLOW}[Warning]  Log file not found at logs/app.log${NC}"
fi
echo ""

# 6. Process Check
echo "6.  Checking Python Process..."
if pgrep -f "uvicorn api.main:app" > /dev/null; then
    echo -e "${GREEN}[OK] Uvicorn process is running${NC}"
    ps aux | grep "uvicorn api.main" | grep -v grep | awk '{print "   - PID: "$2" | CPU: "$3"% | MEM: "$4"%"}'
else
    echo -e "${RED}[Error] Uvicorn process not found${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo " Summary"
echo "=========================================="
echo "Check completed at: $(date)"
echo ""
echo " Tips:"
echo "   - Watch logs: tail -f logs/app.log"
echo "   - Scheduler jobs: curl $API_URL/api/scheduler/health | jq '.data.jobs'"
echo "   - Redis monitor: redis-cli MONITOR"
echo ""
