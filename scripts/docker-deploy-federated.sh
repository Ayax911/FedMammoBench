#!/bin/bash
# Docker deployment script for federated learning experiments
# Usage: ./scripts/docker-deploy-federated.sh exp14 [--clean] [--monitor]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXPERIMENT="${1:-exp14}"
CLEAN_FIRST=true
MONITOR_ROUNDS=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --no-clean)
      CLEAN_FIRST=false
      ;;
    --monitor)
      MONITOR_ROUNDS=true
      ;;
  esac
done

# Validate environment
if [ -z "$REPO" ]; then
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  export REPO
fi

if [ -z "$MAMMO_DATA" ]; then
  MAMMO_DATA="/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench"
  export MAMMO_DATA
fi

# Check config exists
if [ ! -f "$REPO/configs/$EXPERIMENT/server.yaml" ]; then
  echo -e "${RED}✗ Config not found: $REPO/configs/$EXPERIMENT/server.yaml${NC}"
  exit 1
fi

echo -e "${YELLOW}=== Docker Deployment: $EXPERIMENT ===${NC}"
echo "REPO: $REPO"
echo "MAMMO_DATA: $MAMMO_DATA"
echo ""

# Clean previous containers
if [ "$CLEAN_FIRST" = true ]; then
  echo -e "${YELLOW}Cleaning previous containers...${NC}"
  docker rm -f "${EXPERIMENT}_server" "${EXPERIMENT}_client"{1..5} 2>/dev/null || true
fi

# Function to launch server
launch_server() {
  echo -e "${YELLOW}[1/6] Launching server...${NC}"
  docker run -d --name "${EXPERIMENT}_server" --gpus all --network host \
    -v "$REPO/configs:/app/configs:ro" \
    -v "$REPO/weights:/app/weights:ro" \
    -v "$REPO/runs:/app/runs" \
    ayax911/federal-learning:latest \
    python scripts/run_server.py \
      --config "configs/$EXPERIMENT/server.yaml" \
      --address 0.0.0.0:8080

  sleep 3
  if docker logs "${EXPERIMENT}_server" 2>&1 | grep -q "gRPC server running"; then
    echo -e "${GREEN}✓ Server ready${NC}"
    return 0
  else
    echo -e "${RED}✗ Server failed to start${NC}"
    docker logs "${EXPERIMENT}_server" | tail -20
    return 1
  fi
}

# Function to launch client
launch_client() {
  local client_id=$1
  local dataset=$2
  local manifest=$3

  echo -e "${YELLOW}[$(($client_id + 1))/6] Launching client $client_id ($dataset)...${NC}"

  docker run -d --name "${EXPERIMENT}_client$client_id" --gpus all --network host \
    -v "$REPO/configs:/app/configs:ro" \
    -v "$REPO/manifests:/app/manifests:ro" \
    -v "$REPO/weights:/app/weights:ro" \
    -v "$MAMMO_DATA:/app/data:ro" \
    -v "$REPO/runs:/app/runs" \
    ayax911/federal-learning:latest \
    python scripts/run_client.py \
      --config "configs/$EXPERIMENT/client.yaml" \
      --server 127.0.0.1:8080 \
      --client-id "$client_id" \
      --manifest "manifests/$manifest"

  sleep 5
  if docker logs "${EXPERIMENT}_client$client_id" 2>&1 | grep -q "data loaded"; then
    local samples=$(docker logs "${EXPERIMENT}_client$client_id" 2>&1 | grep "train=" | grep -oP 'train=\K\d+' | head -1)
    echo -e "${GREEN}✓ Client $client_id ready ($samples train samples)${NC}"
    return 0
  else
    echo -e "${RED}✗ Client $client_id failed${NC}"
    docker logs "${EXPERIMENT}_client$client_id" | tail -15
    return 1
  fi
}

# Launch all services
launch_server || exit 1

launch_client 1 "CMMD" "cmmd-split.csv" || exit 1
launch_client 2 "InBreast" "inbreast-split.csv" || exit 1
launch_client 3 "CDD-CESM" "cdd-cesm-split.csv" || exit 1
launch_client 4 "KAU-BCMD" "kau-bcmd-split.csv" || exit 1
launch_client 5 "DMID" "dmid-split.csv" || exit 1

# Verify all containers running
echo ""
echo -e "${YELLOW}=== Container Status ===${NC}"
docker ps | grep "$EXPERIMENT" || echo "No containers found"

# Monitor if requested
if [ "$MONITOR_ROUNDS" = true ]; then
  echo ""
  echo -e "${YELLOW}Monitoring ROUND 1...${NC}"
  until docker logs "${EXPERIMENT}_server" 2>&1 | grep -q "\[ROUND 2\]"; do
    sleep 5
  done
  echo -e "${GREEN}✓ ROUND 1 completed${NC}"

  docker logs "${EXPERIMENT}_server" 2>&1 | grep -E "\[ROUND 1\]|loss=|auc=" | tail -5
fi

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Monitor logs:"
echo "  docker logs -f ${EXPERIMENT}_server"
echo ""
echo "Stop all containers:"
echo "  docker stop ${EXPERIMENT}_server ${EXPERIMENT}_client{1..5}"
