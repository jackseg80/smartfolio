#!/usr/bin/env bash
set -euo pipefail

# Docker Complete Cleanup Script for SmartFolio
# WARNING: This will remove ALL Docker containers, images, and optionally volumes

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🧹 Docker Complete Cleanup - SmartFolio${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: This will remove:${NC}"
echo "   • All SmartFolio containers (running and stopped)"
echo "   • All SmartFolio Docker images"
echo "   • All unused Docker networks"
echo "   • Optionally: Docker volumes (data, cache, logs)"
echo ""
echo -e "${YELLOW}Your local files (data/, logs/, cache/) will NOT be affected${NC}"
echo ""

read -p "Continue with cleanup? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}✅ Cleanup cancelled${NC}"
    exit 0
fi

# Step 1: Stop all SmartFolio containers
echo ""
echo -e "${YELLOW}📦 Step 1/5: Stopping containers...${NC}"

# Stop production containers
docker-compose -f docker-compose.prod.yml down 2>/dev/null || echo "  (no prod containers)"

# Stop dev containers
docker-compose -f docker-compose.yml down 2>/dev/null || echo "  (no dev containers)"

# Stop legacy containers
docker stop smartfolio_api_1 smartfolio-api smartfolio-redis crypto-rebal 2>/dev/null || true

echo -e "${GREEN}✅ Containers stopped${NC}"

# Step 2: Remove all SmartFolio containers
echo ""
echo -e "${YELLOW}🗑️  Step 2/5: Removing containers...${NC}"

docker rm -f smartfolio_api_1 smartfolio-api smartfolio-redis crypto-rebal 2>/dev/null || true
docker container prune -f

echo -e "${GREEN}✅ Containers removed${NC}"

# Step 3: Remove SmartFolio images
echo ""
echo -e "${YELLOW}🖼️  Step 3/5: Removing images...${NC}"

# Remove images by name pattern
docker images | grep smartfolio | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true
docker images | grep crypto-rebal | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true

# Remove dangling images
docker image prune -f

echo -e "${GREEN}✅ Images removed${NC}"

# Step 4: Remove networks
echo ""
echo -e "${YELLOW}🌐 Step 4/5: Removing networks...${NC}"

docker network rm smartfolio-network smartfolio-net crypto-net 2>/dev/null || true
docker network prune -f

echo -e "${GREEN}✅ Networks removed${NC}"

# Step 5: Optionally remove volumes
echo ""
echo -e "${YELLOW}💾 Step 5/5: Remove Docker volumes?${NC}"
echo -e "${RED}⚠️  This will delete Redis data (alerts, cache, streams)${NC}"
echo -e "${YELLOW}   Local files in data/, logs/, cache/ are safe${NC}"
echo ""

read -p "Remove Docker volumes? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker volume rm smartfolio-redis-data redis_data 2>/dev/null || true
    docker volume prune -f
    echo -e "${GREEN}✅ Volumes removed${NC}"
else
    echo -e "${YELLOW}⏭️  Volumes kept${NC}"
fi

# Final cleanup
echo ""
echo -e "${YELLOW}🧹 Final cleanup...${NC}"
docker system prune -f

# Summary
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Docker Cleanup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}📊 Current Docker Status:${NC}"
docker ps -a
echo ""
docker images
echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo "   1. Rebuild from scratch: ./deploy.sh"
echo "   2. Or start dev mode:    docker-compose up -d --build"
echo ""
