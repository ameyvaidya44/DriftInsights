# DriftInsights — System Architecture

> Explainable and Adaptive ML System for Concept Drift Detection, Diagnosis, and Adaptation

---

## Overview

DriftInsights is a production-grade, real-time ML system that detects **concept drift** in streaming data, explains *why* it happened using SHAP, and automatically adapts the model. It has three major layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│   Dashboard │ LiveStream │ DriftAnalytics │ Explainability      │
│   Models    │ Pipeline   │ Admin Pages                          │
└────────────────────────┬────────────────────────────────────────┘
                         │  REST (HTTP) + WebSocket (ws://)
┌────────────────────────▼────────────────────────────────────────┐
│                     API LAYER (FastAPI)                         │
│   Routes: /predict /metrics /drift /models /xai /pipeline      │
│   WebSocket: /ws  │  Background Tasks  │  PostgreSQL DB         │
└────────────────────────┬────────────────────────────────────────┘
                         │  Python calls
┌────────────────────────▼────────────────────────────────────────┐
│                   CORE ML PIPELINE (src/)                       │
│                                                                 │
│  DataHandler → BaseModel → DriftDetector → XAIEngine           │
│                    ↓                           ↓                │
│             MetricsTracker          DecisionEngine (FSM)        │
│                                           ↓                     │
│                                   AdaptationModule              │
│                                           ↓                     │
│                                    EventLogger                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Core ML Pipeline (`src/`)

This is the brain of the system. All ML logic lives here.

### `config.py` — Configuration
Loads `config/default.yaml` and exposes it with dot-notation access (e.g., `config.drift.adwin.delta`). Controls every tunable parameter: dataset, model type, ADWIN sensitivity, SHAP window sizes, adaptation thresholds, and cooldown periods.

### `data_handler.py` — Data Loading & Streaming
Loads datasets (SEA synthetic or ELEC2 real-world), preprocesses them, and splits into train/val/test. During streaming, it feeds samples one-by-one to simulate a live data stream — making the system behave like a real production ML pipeline.

### `base_model.py` — Model Wrapper
Wraps XGBoost, LightGBM, or Random Forest behind a unified interface. Key feature: `hot_swap()` — replaces the live model with a newly trained one atomically, with zero downtime. Handles model serialization to `models/` via joblib.

### `drift_detector.py` — ADWIN Drift Detection
Uses the **ADWIN** (Adaptive Windowing) algorithm from the `river` library. ADWIN monitors the error stream and detects statistically significant distribution shifts. When the error rate in a recent window diverges from the historical window, it fires a `DriftEvent` with metadata: sample index, pre/post error rates, and ADWIN window width. Configurable sensitivity (`delta`), minimum gap between detections, and warmup period.

### `xai_engine.py` — SHAP Explainability
When drift is detected, this engine answers *why*. It uses **SHAP TreeExplainer** to compute feature importance for a pre-drift window and a post-drift window, then calculates the shift:

```
ΔE_i     = |mean_SHAP_before_i - mean_SHAP_after_i|   (per feature)
ΔE_total = L2 norm of all ΔE_i                         (overall drift magnitude)
```

Output is a `DriftExplanation` with ranked features, direction of change (increase/decrease), and distribution statistics (mean, std) for each feature before and after drift.

### `decision_engine.py` — FSM Decision Engine
A **Finite State Machine** with three states:

```
IDLE ──(drift detected)──► ADAPTING ──(done)──► COOLDOWN ──(timer)──► IDLE
```

Maps drift severity (derived from `ΔE_total`) to an adaptation strategy:

| Severity | ΔE_total Range | Strategy |
|---|---|---|
| Minor | < moderate_threshold | REWEIGHT |
| Moderate | < severe_threshold | INCREMENTAL_RETRAIN |
| Severe | ≥ severe_threshold | FULL_RETRAIN |

The cooldown prevents adaptation thrashing. Every decision includes a human-readable `reason` string for auditability.

### `adaptation_module.py` — Model Adaptation
Executes the strategy chosen by the Decision Engine:

- **REWEIGHT** — Applies exponential decay weights to training samples, favoring recent data. Fast, no retraining.
- **INCREMENTAL_RETRAIN** — Retrains on the most recent N samples using `warm_start`. Quick and targeted.
- **FULL_RETRAIN** — Trains a brand new model from scratch on all available data. Thorough but slower.

After adaptation, a **validation gate** checks if the new model's accuracy meets the threshold (with tolerance). If it fails, the old model is **rolled back** and the event is logged. Implements semantic versioning (`v1.0.0`, `v1.0.1`, ...) for every model deployed.

### `metrics_tracker.py` — Metrics
Tracks running accuracy, windowed error rate, and other performance metrics as samples stream through. Provides snapshots for the API.

### `event_logger.py` — Structured Logging
Writes all events (drift detections, adaptations, rollbacks, FSM transitions) to JSONL files in `logs/`. Each session gets its own file (`session_<timestamp>_<id>.jsonl`). This is the full audit trail.

### `pipeline.py` — Orchestrator
The main coordinator. Wires all components together and runs the three-phase lifecycle:

```
1. setup()       → Initialize all components from config
2. train()       → Train base model on training split
3. run_stream()  → Per-sample loop:
                     predict → update metrics → detect drift
                     → explain (SHAP) → decide (FSM) → adapt
                     → validate → hot_swap or rollback
                     → broadcast via WebSocket → log event
```

---

## Layer 2 — API Layer (`api/`)

Exposes the pipeline to the outside world via HTTP and WebSocket.

### `api/main.py` — FastAPI App
The entrypoint. On startup, it auto-initializes the pipeline (setup + train), injects the WebSocket manager, and registers all routes. Uses FastAPI's `lifespan` context for clean startup/shutdown. CORS is configured to allow the frontend at `localhost:5173`.

### `api/routes/` — REST Endpoints

| Route File | Endpoints |
|---|---|
| `predict` | `POST /predict` — run a single prediction |
| `metrics` | `GET /metrics`, `/metrics/history`, `/metrics/snapshot` |
| `drift` | `GET /drift/events`, `/drift/latest`, `/drift/events/{id}/explanation` |
| `models` | `GET /models`, `/models/history` |
| `xai` | `GET /xai/latest`, `/xai/history`, `/xai/summary` |
| `pipeline` | `POST /pipeline/start`, `GET /pipeline/status/fsm` |

### `api/websocket.py` — Real-Time Push
A `ConnectionManager` that maintains active WebSocket connections. The pipeline calls `manager.broadcast()` after every significant event (drift detected, adaptation completed, metrics update), pushing JSON payloads to all connected frontend clients in real time.

### `api/database.py` — Persistence
SQLAlchemy models backed by PostgreSQL. Stores `DriftEventRecord`, `MetricsSnapshotRecord`, `AdaptationRecord`, and `XAIRecord`. DB writes are non-critical — failures are caught and logged, never crashing the pipeline.

### `api/tasks.py` — Background Tasks
Manages long-running pipeline operations (like `run_stream()`) in background threads so the API stays responsive. The `POST /pipeline/start` endpoint triggers streaming here without blocking.

### `api/schemas.py` — Pydantic Models
Request/response schemas for all API endpoints. Ensures type safety and auto-generates the OpenAPI docs at `/docs`.

---

## Layer 3 — Frontend (`frontend/`)

A React + Vite SPA with Recharts for visualization.

### `src/services/api.js` — API Client
Axios-based client pointing to `localhost:8000`. Exports typed fetch functions for every endpoint. `fetchAllDashboardData()` fires 6 parallel requests for the main dashboard load.

### `src/hooks/useWebSocket.js` — WebSocket Hook
Maintains a persistent WebSocket connection to `ws://localhost:8000/ws`. Exposes `connected`, `lastUpdate`, and `connectionFailed` state to components. Drives real-time UI updates.

### `src/context/AuthContext.jsx` — Auth
Provides login state and role (`user` vs `admin`) via React Context. `RequireAuth` and `RequireAdmin` route guards wrap protected pages.

### Pages

| Page | Purpose |
|---|---|
| `Dashboard` | Overview: metrics cards, live decision feed, pipeline flow viz, system health |
| `LiveStream` | Real-time sample-by-sample prediction feed |
| `DriftAnalytics` | Drift event timeline and frequency charts |
| `Explainability` | SHAP feature importance charts, per-event explanations |
| `Models` | Model version history, accuracy over time |
| `Pipeline` | FSM state visualization, pipeline controls (start/stop) |
| `Admin/Overview` | System-wide stats and health summary |
| `Admin/Users` | User management and role assignment |
| `Admin/Config` | Live config editor |
| `Admin/Datasets` | Dataset browser |
| `Admin/Models` | Model registry and version management |
| `Admin/Audit` | Full audit log viewer |
| `Admin/SystemHealth` | Infrastructure health indicators |

---

## Data Flow — End to End

```
Raw Data (SEA / ELEC2)
       │
       ▼
  DataHandler ──splits──► train / val / test
       │
       ▼
  BaseModel.fit(X_train)
       │
       ▼  (streaming loop — one sample at a time)
  BaseModel.predict(x_i)
       │
       ├──► MetricsTracker.update(error_i)
       │
       ▼
  DriftDetector.update(error_i)
       │
       │  (if drift detected)
       ▼
  XAIEngine.explain(pre_window, post_window)
       │  → ΔE_total, top_features, direction
       ▼
  DecisionEngine.evaluate(ΔE_total)
       │  → FSM transition + AdaptationCommand
       ▼
  AdaptationModule.execute(command)
       │  → new model trained
       │  → validation gate check
       │  → hot_swap  OR  rollback
       ▼
  EventLogger.log(event)  +  WebSocket.broadcast(event)
       │                              │
       ▼                              ▼
  logs/*.jsonl                  Frontend (React)
  PostgreSQL DB                 real-time UI update
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML Core | XGBoost, LightGBM, scikit-learn, SHAP, river (ADWIN) |
| Backend | FastAPI, SQLAlchemy, PostgreSQL, Uvicorn |
| Frontend | React, Vite, Recharts, Axios, React Router |
| Config | YAML (`config/default.yaml`) |
| Logging | JSONL structured logs per session (`logs/`) |
| Testing | Pytest (`tests/`) |
| Models | joblib serialization, semantic versioning (`models/`) |

---

## Project Structure

```
driftinsights/
├── config/
│   └── default.yaml          # All system parameters
├── src/
│   ├── config.py             # YAML config loader (dot-notation)
│   ├── data_handler.py       # Data loading, preprocessing, streaming
│   ├── base_model.py         # XGBoost/LightGBM/RF wrapper + hot_swap
│   ├── drift_detector.py     # ADWIN drift detection
│   ├── xai_engine.py         # SHAP explanation engine (ΔE)
│   ├── decision_engine.py    # FSM + severity → strategy mapping
│   ├── adaptation_module.py  # Reweight / Incremental / Full retrain
│   ├── metrics_tracker.py    # Running + windowed metrics
│   ├── event_logger.py       # JSONL structured logging
│   ├── pipeline.py           # Main orchestrator
│   └── utils/helpers.py      # Shared utilities
├── api/
│   ├── main.py               # FastAPI app + lifespan
│   ├── routes/               # predict, drift, models, metrics, xai
│   ├── database.py           # SQLAlchemy + PostgreSQL models
│   ├── websocket.py          # Real-time broadcast manager
│   ├── tasks.py              # Background task manager
│   └── schemas.py            # Pydantic request/response models
├── frontend/
│   └── src/
│       ├── pages/            # Dashboard, LiveStream, Drift, XAI, Models, Pipeline, Admin
│       ├── components/       # Reusable UI components + charts
│       ├── hooks/            # useWebSocket, useFetch
│       ├── services/api.js   # Axios API client
│       └── context/          # AuthContext
├── models/                   # Saved model files + version_history.json
├── logs/                     # Per-session JSONL + .log files
├── tests/                    # Pytest test suite
└── main.py                   # CLI entrypoint
```
