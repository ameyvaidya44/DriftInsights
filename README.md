# DriftInsights

**Explainable and Adaptive ML System for Concept Drift Detection, Diagnosis, and Adaptation**

A production-grade, real-time ML pipeline that:
- **Detects** concept drift using ADWIN statistical methods
- **Explains** drift using SHAP-based explanation shift (delta-E)
- **Adapts** models automatically using intelligent retraining strategies

## Architecture

```
Data -> Model -> Prediction -> Error
  -> Drift Detection (ADWIN) -> XAI (SHAP) -> Decision Engine (FSM)
  -> Adaptation -> Model Update (hot_swap)
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the Pipeline (CLI)

```bash
python main.py                            # Default: SEA + XGBoost
python main.py --dataset elec2            # Use ELEC2 dataset
python main.py --model lightgbm           # Use LightGBM
python main.py --max-samples 5000         # Limit streaming
```

### 3. Start the API Server

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs at: http://localhost:8000/docs

### 4. Start the Dashboard

```bash
cd frontend
npm install
npm run dev
```

Dashboard at: http://localhost:5173

## Project Structure

```
driftinsights/
  config/default.yaml          # System configuration
  src/
    config.py                  # YAML config with dot-notation
    data_handler.py            # Data loading + preprocessing + streaming
    base_model.py              # XGBoost/LightGBM/RF wrapper + hot_swap
    drift_detector.py          # ADWIN drift detection
    xai_engine.py              # SHAP explanation engine
    decision_engine.py         # FSM + severity -> strategy mapping
    adaptation_module.py       # Reweight/Incremental/Full retrain
    metrics_tracker.py         # Running + windowed metrics
    event_logger.py            # JSONL structured logging
    pipeline.py                # Main orchestrator
  api/
    main.py                    # FastAPI app
    routes/                    # REST endpoints
    database.py                # SQLAlchemy + PostgreSQL
    websocket.py               # Real-time event push
    tasks.py                   # Background task manager
  frontend/                    # React + Recharts dashboard
  tests/                       # Pytest test suite
  main.py                      # CLI entrypoint
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/status` | GET | Pipeline status |
| `/predict` | POST | Make prediction |
| `/metrics` | GET | Current metrics |
| `/metrics/history` | GET | Metrics history |
| `/drift/events` | GET | Drift event list |
| `/drift/latest` | GET | Latest drift event |
| `/models` | GET | Model info |
| `/models/history` | GET | Adaptation history |
| `/pipeline/start` | POST | Start streaming |
| `/ws` | WebSocket | Real-time events |

## Configuration

All parameters are in `config/default.yaml`:
- **Data**: dataset, split ratios, streaming config
- **Model**: type, hyperparameters, save directory
- **Drift**: ADWIN delta, min gap, warmup
- **XAI**: SHAP samples, window size, top-k features
- **Decision**: severity thresholds, cooldown
- **Adaptation**: strategies, validation threshold

## Performance

| Metric | Target | Achieved |
|---|---|---|
| Training Accuracy | >= 0.80 | 0.90 |
| Drift Detection | < 100 samples | Yes |
| SHAP Runtime | < 30s | < 1s |
| Incremental Retrain | < 10s | < 0.1s |

## License

MIT
