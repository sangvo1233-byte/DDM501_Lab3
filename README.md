# Lab 3: Testing & CI/CD for ML Systems

![CI Pipeline](https://github.com/sangvo1233-byte/DDM501_Lab3/actions/workflows/ci.yml/badge.svg)

## Overview

Implement comprehensive testing strategies and CI/CD pipelines for the movie rating prediction system to ensure quality and automate deployment.

**Course:** DDM501 - AI in Production: From Models to Systems  
**Weight:** 15% of total grade  
**Duration:** 3 hours (in-class) + 1 week to complete  
**Prerequisites:** Lab 1 and Lab 2 completed

## Learning Objectives

- Write comprehensive unit tests for ML components
- Implement integration tests for API endpoints
- Create data validation tests
- Design model behavioral tests (invariance, directional, minimum functionality)
- Set up CI/CD pipelines with GitHub Actions
- Implement automated code quality checks

## Project Structure

```
ddm501-lab3-starter/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── model.py            # ML model class
│   ├── schemas.py          # Pydantic schemas
│   └── config.py           # Configuration
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_model.py   # Model unit tests (TODO)
│   │   └── test_schemas.py # Schema tests (TODO)
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_api.py     # API tests (TODO)
│   ├── data/
│   │   ├── __init__.py
│   │   └── test_data_quality.py  # Data tests (TODO)
│   └── model/
│       ├── __init__.py
│       └── test_model_behavior.py  # Behavioral tests (TODO)
├── .github/
│   └── workflows/
│       ├── ci.yml          # CI pipeline (TODO)
│       └── cd.yml          # CD pipeline (TODO)
├── scripts/
│   └── train_model.py      # Model training script
├── models/                 # Saved models
├── .pre-commit-config.yaml # Pre-commit hooks (TODO)
├── pyproject.toml          # Project configuration
├── requirements.txt
├── requirements-dev.txt    # Development dependencies
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/[your-repo]/ddm501-lab3-starter.git
cd ddm501-lab3-starter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Train Model (if not exists)

```bash
python scripts/train_model.py
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/data/ -v
pytest tests/model/ -v
```

### 4. Code Quality Checks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run all checks manually
pre-commit run --all-files

# Individual tools
black app/ tests/
flake8 app/ tests/
mypy app/
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## TODO Tasks

Complete the following files:

### Test Files
- [x] `tests/unit/test_model.py` - Unit tests for model class
- [x] `tests/unit/test_schemas.py` - Schema validation tests
- [x] `tests/integration/test_api.py` - API endpoint tests
- [x] `tests/data/test_data_quality.py` - Data quality tests
- [x] `tests/model/test_model_behavior.py` - Behavioral tests

### CI/CD Files
- [x] `.github/workflows/ci.yml` - CI pipeline
- [x] `.github/workflows/cd.yml` - CD pipeline (BONUS)
- [x] `.pre-commit-config.yaml` - Pre-commit hooks

## Results

| Check | Result |
|---|---|
| Tests | **95 passed**: unit (model + schemas), integration (API), data quality, model behaviour |
| Coverage | **98.5%** of `app/` (CI enforces a minimum of 80% with `--cov-fail-under=80`) |
| flake8 / black / isort | clean (line length 100) |
| mypy | `Success: no issues found in 5 source files` |
| Model | SVD on MovieLens 100K, 5-fold CV RMSE ~0.94 |

### What was implemented

- **Unit tests**: prediction type and range, batch length and order, `is_loaded`, graceful handling of
  `None` and empty ids, a missing or corrupt model file, the unloaded-model guard, and singleton reset.
- **Schema tests**: missing and empty fields, whitespace-only ids, `None`, int vs str (Pydantic v2 does not
  coerce an int into a `str` field), rating bounds 1.0 to 5.0 including the boundaries, and batch sizes 0 and 101.
- **Integration tests**: every endpoint's status and shape, 422 on invalid JSON, 404 and 405, and a
  *degraded service* class that monkeypatches the model to check the 503 (model missing) and 500 (model
  raises) paths.
- **Data tests**: rating range, ids, completeness, distribution (mean, std, distinct values), uniqueness, types.
- **Behavioural tests**: invariance (determinism, batch order, single vs batch), directional (different
  users or movies get different scores, errors are bounded on known ratings), minimum functionality, and
  unknown users or movies falling back to the global mean.
- **CI** (`.github/workflows/ci.yml`): `lint` and `type-check` run in parallel; `test` trains the model, runs
  pytest with an 80% coverage gate and uploads the coverage report and model artifact; `build` builds the Docker
  image with the trained model and smoke-tests `/health` and `/predict`. pip and the MovieLens download are
  cached.
- **CD** (`.github/workflows/cd.yml`, bonus): on a `v*` tag it builds and pushes to GHCR (using `GITHUB_TOKEN`,
  so no secrets are needed), creates a GitHub Release, deploys to `staging` with a smoke test, then `production`
  (add required reviewers to the environment for a manual approval gate). **This part is a demonstration:** the
  image build and push, the release and the staging smoke test are real steps, but the production job only
  `echo`s, because the course provides no production target. Replace that step with the real deploy command
  (`kubectl set image`, `gcloud run deploy` and so on) when one exists.
- **Pre-commit**: hygiene hooks, black, isort, flake8, mypy, and fast unit tests.

### Fixes to the starter

- `tests/conftest.py`: the `test_client` fixture now uses `with TestClient(app)`, so FastAPI's startup event
  runs and the model is loaded. A bare `TestClient(app)` skips startup, and every prediction test got a 503.
- `requirements.txt`: `scikit-surprise` was bumped from 1.1.3 to **1.1.4**, which builds against current numpy and
  Python 3.12.
- `Dockerfile`: `python:3.10-slim` has no `curl`, so the original `HEALTHCHECK` could never pass. The check now
  uses the Python standard library and requires `model_loaded: true`. A compiler is installed for
  scikit-surprise, and the container runs as a non-root user.

## Test Types

### Unit Tests
Test individual functions and classes in isolation.

```python
def test_model_loads_successfully(model):
    assert model.is_loaded()
```

### Integration Tests
Test component interactions and API endpoints.

```python
def test_predict_valid_request(test_client):
    response = test_client.post("/predict", json={"user_id": "196", "movie_id": "242"})
    assert response.status_code == 200
```

### Data Tests
Validate data quality and schema.

```python
def test_ratings_in_valid_range(sample_ratings):
    for r in sample_ratings:
        assert 1.0 <= r["rating"] <= 5.0
```

### Behavioral Tests
Test model behavior patterns.

```python
def test_same_input_same_output(model):
    result1 = model.predict("196", "242")
    result2 = model.predict("196", "242")
    assert result1 == result2
```

## CI/CD Pipeline

### Continuous Integration
- Runs on every push and pull request
- Executes linting, type checking, and tests
- Reports code coverage

### Continuous Deployment (BONUS)
- Triggered on version tags
- Builds and pushes Docker image
- Deploys to staging/production

## Grading Rubric

| Criteria | Weight |
|----------|--------|
| Test Coverage (unit, integration, data, model) | 30% |
| CI/CD Pipeline | 30% |
| Code Quality | 20% |
| Documentation | 20% |

**Minimum Requirements:**
- 80% code coverage
- All CI checks passing
- Pre-commit hooks configured

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [pre-commit](https://pre-commit.com/)
- [Black](https://black.readthedocs.io/)
- [Flake8](https://flake8.pycqa.org/)
- [mypy](https://mypy.readthedocs.io/)

## Submission

1. Complete all TODO items
2. Ensure all tests pass
3. Achieve minimum 80% coverage
4. Push to GitHub with CI badge
5. Submit repository link via LMS

## License

MIT License - For educational purposes only.
