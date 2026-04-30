# Test data for the ML API

Pre-canned payloads for hitting the FastAPI service in `ML models/api.py`.
Start the API first:

```bash
cd "ML models"
uvicorn api:app --port 8003
```

## Files

| File | Use with |
|---|---|
| `demand_single.json` | `POST /predict/demand` — single fast-mover example |
| `dead_stock_single.json` | `POST /predict/dead-stock` — single idle-stock example |
| `batch_codes.json` | `POST /predict/batch` — list of real product codes |
| `test_products.json` | 15 hand-crafted products with `expected_demand` / `expected_dead` for end-to-end smoke testing |
| `run_test_suite.py` | Runs all 15 products through the API and prints accuracy / mismatches |

## Examples

### Single demand prediction
```bash
curl -X POST http://localhost:8003/predict/demand \
  -H "Content-Type: application/json" \
  -d @test_data/demand_single.json
```

### Single dead-stock prediction
```bash
curl -X POST http://localhost:8003/predict/dead-stock \
  -H "Content-Type: application/json" \
  -d @test_data/dead_stock_single.json
```

### Batch by product code
```bash
curl -X POST http://localhost:8003/predict/batch \
  -H "Content-Type: application/json" \
  -d @test_data/batch_codes.json
```

### Lookup a single product
```bash
curl http://localhost:8003/predict/by-code/10000019
```

### Run all 15 hand-crafted products against the API
```bash
python test_data/run_test_suite.py
# point at a different host:
python test_data/run_test_suite.py --base-url http://localhost:8003
```

## Swagger UI

The interactive docs at `http://localhost:8003/docs` accept any of these
JSON payloads — paste them into the "Try it out" body for each endpoint.
