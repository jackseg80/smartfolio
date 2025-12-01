#!/usr/bin/env bash
set -euo pipefail

# SmartFolio Production Deployment Script
# Usage: ./deploy.sh [--force] [--skip-build]
#
# Options:
#   --force       Force git reset (écrase changements locaux sans demander)
#   --skip-build  Skip Docker rebuild (restart seulement)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FORCE_RESET=0
SKIP_BUILD=0

# Parse arguments
for arg in "$@"; do
    case $arg in
        --force) FORCE_RESET=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        -h|--help)
            head -n 10 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🚀 SmartFolio Production Deployment${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ Error: docker-compose.prod.yml not found${NC}"
    echo "   Please run this script from the smartfolio root directory"
    exit 1
fi

# Step 1: Check for local changes
echo -e "${YELLOW}📋 Step 1/5: Checking for local changes...${NC}"
if git diff --quiet && git diff --cached --quiet; then
    echo -e "${GREEN}✅ No local changes${NC}"
else
    echo -e "${YELLOW}⚠️  Local changes detected:${NC}"
    git status --short

    if [ $FORCE_RESET -eq 0 ]; then
        echo ""
        echo -e "${YELLOW}Do you want to discard local changes and pull from GitHub?${NC}"
        echo -e "${RED}This will erase any uncommitted changes!${NC}"
        read -p "Continue? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}❌ Deployment cancelled${NC}"
            exit 1
        fi
    fi

    # Save changes just in case
    BACKUP_FILE="/tmp/smartfolio_backup_$(date +%Y%m%d_%H%M%S).patch"
    git diff > "$BACKUP_FILE"
    echo -e "${GREEN}💾 Local changes backed up to: $BACKUP_FILE${NC}"

    # Reset to origin
    git reset --hard origin/main
    echo -e "${GREEN}✅ Local changes discarded${NC}"
fi

# Step 2: Pull latest version
echo ""
echo -e "${YELLOW}📥 Step 2/5: Pulling latest version from GitHub...${NC}"
git fetch origin main
git pull origin main
echo -e "${GREEN}✅ Latest version pulled ($(git log -1 --format='%h - %s'))${NC}"

# Step 3: Verify and update .env for production
echo ""
echo -e "${YELLOW}🔧 Step 3/6: Checking .env configuration...${NC}"

# Detect server IP
SERVER_IP=$(hostname -I | awk '{print $1}')
if [ -z "$SERVER_IP" ]; then
    echo -e "${RED}❌ Error: Could not detect server IP${NC}"
    exit 1
fi

# Check if API_BASE_URL needs updating
CURRENT_API_URL=$(grep "^API_BASE_URL=" .env 2>/dev/null | cut -d'=' -f2 || true)
EXPECTED_API_URL="http://${SERVER_IP}:8080"

if [ "$CURRENT_API_URL" != "$EXPECTED_API_URL" ]; then
    echo -e "${YELLOW}⚠️  API_BASE_URL mismatch:${NC}"
    echo "   Current:  $CURRENT_API_URL"
    echo "   Expected: $EXPECTED_API_URL"

    # Update .env file
    if grep -q "^API_BASE_URL=" .env 2>/dev/null || false; then
        sed -i "s|^API_BASE_URL=.*|API_BASE_URL=${EXPECTED_API_URL}|" .env
    else
        echo "API_BASE_URL=${EXPECTED_API_URL}" >> .env
    fi

    echo -e "${GREEN}✅ Updated API_BASE_URL to: $EXPECTED_API_URL${NC}"
else
    echo -e "${GREEN}✅ API_BASE_URL already correct: $CURRENT_API_URL${NC}"
fi

# Step 4: Verify price cache exists
echo ""
echo -e "${YELLOW}💰 Step 4/6: Verifying price cache...${NC}"
CACHE_COUNT=$(find data/price_history -name "*.json" 2>/dev/null | wc -l)
if [ "$CACHE_COUNT" -lt 100 ]; then
    echo -e "${RED}⚠️  Warning: Only $CACHE_COUNT price cache files found (expected ~127)${NC}"
    echo -e "${YELLOW}   Price history may be incomplete. Run: scp from Windows or update_price_history.py${NC}"
else
    echo -e "${GREEN}✅ Price cache OK: $CACHE_COUNT files${NC}"
fi

# Step 5: Docker rebuild or restart
echo ""
if [ $SKIP_BUILD -eq 1 ]; then
    echo -e "${YELLOW}🔄 Step 5/6: Restarting Docker (skip build)...${NC}"
    docker-compose restart
else
    echo -e "${YELLOW}🐳 Step 5/6: Rebuilding and restarting Docker...${NC}"

    # Stop old containers
    docker-compose down

    # Clean orphaned containers from non-prod compose
    docker stop smartfolio_api_1 2>/dev/null || true
    docker rm smartfolio_api_1 2>/dev/null || true

    # Build and start
    docker-compose up -d --build
fi

echo -e "${GREEN}✅ Docker containers started${NC}"

# Step 6: Health check
echo ""
echo -e "${YELLOW}🏥 Step 6/6: Waiting for services to be healthy...${NC}"
sleep 10

# Check Docker containers
if docker ps | grep -q "smartfolio-api"; then
    echo -e "${GREEN}✅ Container smartfolio-api: running${NC}"
else
    echo -e "${RED}❌ Container smartfolio-api: not found${NC}"
    docker-compose logs --tail 50
    exit 1
fi

# Check API health
echo -n "   Testing API endpoint... "
if curl -sf http://localhost:8080/docs > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    echo -e "${YELLOW}   API may still be starting up. Check logs with:${NC}"
    echo -e "   docker-compose logs -f"
fi

# Check scheduler
echo -n "   Testing scheduler... "
SCHEDULER_STATUS=$(curl -sf http://localhost:8080/api/scheduler/health 2>/dev/null | grep -o '"enabled":[^,]*' | cut -d':' -f2)
if [ "$SCHEDULER_STATUS" == "true" ]; then
    echo -e "${GREEN}✅ Enabled${NC}"
else
    echo -e "${YELLOW}⚠️  Unknown${NC}"
fi

# Final summary
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}📊 Quick Access:${NC}"
echo "   • Dashboard:  http://$(hostname -I | awk '{print $1}'):8080/dashboard.html"
echo "   • API Docs:   http://$(hostname -I | awk '{print $1}'):8080/docs"
echo "   • Risk:       http://$(hostname -I | awk '{print $1}'):8080/risk-dashboard.html"
echo ""
echo -e "${YELLOW}🔍 Useful Commands:${NC}"
echo "   • View logs:        docker-compose logs -f"
echo "   • Check status:     docker-compose ps"
echo "   • Restart:          ./deploy.sh --skip-build"
echo "   • Full redeploy:    ./deploy.sh --force"
echo ""
