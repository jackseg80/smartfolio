import asyncio
import json
from services.ml.orchestrator import get_orchestrator

async def test():
    orch = get_orchestrator()
    preds = await orch.get_unified_predictions(['BTC', 'ETH', 'SOL'], [1,7,30])
    correlation_data = preds.get('models', {}).get('correlation', {})
    print("=== CORRELATION DATA FROM ORCHESTRATOR ===")
    print(json.dumps(correlation_data, indent=2))
    print("\n=== FULL MODELS DATA ===")
    print(json.dumps(list(preds.get('models', {}).keys()), indent=2))

asyncio.run(test())
