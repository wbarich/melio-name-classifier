# Testing Guide

This document explains how to run tests for the Name Classifier project.

## Test Structure

The project includes two types of tests:

1. **Unit Tests** (`test_model.py`) - Test individual components of the NameClassifier model
2. **Integration Tests** (`test_api.py`) - Test the HTTP API endpoints (requires running server)

```
src/tests/
├── __init__.py
├── conftest.py          # Pytest fixtures and test data
├── test_model.py        # Unit tests for the model
└── test_api.py          # Integration tests for the API
```

## Prerequisites

Install test dependencies:

```bash
pip install -r requirements.txt
```

This installs:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-asyncio` - Async test support
- `httpx` - HTTP client for API tests

## Running Tests

### Run All Unit Tests

```bash
python3 -m pytest src/tests/test_model.py -v
```

**Expected output:**
```
============================= test session starts ==============================
...
src/tests/test_model.py::TestNameClassifierInitialization::test_model_initialization PASSED
src/tests/test_model.py::TestNameClassifierInitialization::test_model_has_required_methods PASSED
...
============================== 19 passed in 0.20s ===============================
```

### Run Specific Test Class

```bash
# Test only preprocessing
python3 -m pytest src/tests/test_model.py::TestPreprocess -v

# Test only prediction
python3 -m pytest src/tests/test_model.py::TestPredict -v

# Test end-to-end pipeline
python3 -m pytest src/tests/test_model.py::TestEndToEnd -v
```

### Run Specific Test

```bash
python3 -m pytest src/tests/test_model.py::TestPredict::test_predict_returns_valid_category -v
```

### Run Tests with Coverage Report

```bash
python3 -m pytest src/tests/test_model.py --cov=src --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`. Open it in your browser:

```bash
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
```

### Run Tests with Detailed Output

```bash
# Show print statements
python3 -m pytest src/tests/test_model.py -v -s

# Show local variables on failure
python3 -m pytest src/tests/test_model.py -v -l

# Show full tracebacks
python3 -m pytest src/tests/test_model.py -v --tb=long
```

## Integration Tests (API)

The API integration tests require the server to be running.

### Step 1: Start the Server

In one terminal:

```bash
# Option 1: Run directly
cd src && python3 server.py

# Option 2: Use Docker
docker-compose up
```

Wait for the server to be ready:
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Step 2: Run API Tests

In another terminal:

```bash
python3 -m pytest src/tests/test_api.py -v
```

**Note:** If the server isn't running, these tests will fail with connection errors.

## Test Fixtures

The project uses pytest fixtures to provide reusable test data. See `src/tests/conftest.py` for available fixtures:

### Model Fixtures
- `model` - NameClassifier instance
- `infer_request_factory` - Function to create InferRequest objects
- `sample_infer_request` - Pre-made InferRequest

### Data Fixtures
- `sample_person_names` - List of person names
- `sample_company_names` - List of company names
- `sample_university_names` - List of university names
- `all_sample_names` - Combined dict of all samples
- `edge_case_names` - Names with special characters, edge cases
- `valid_categories` - List of valid classification categories

### Using Fixtures in Tests

```python
def test_my_feature(model, sample_person_names):
    """Test using fixtures."""
    for name in sample_person_names:
        result = model.predict({"name": name})
        assert result["classification"] in ["Person", "Company", "University"]
```

## Test Coverage

Current test coverage for `model.py`:

```
Name         Stmts   Miss  Cover   Missing
------------------------------------------
src/model.py    36      0   100%
```

The model has **100% test coverage**, ensuring all code paths are tested.

## Writing New Tests

### Unit Test Template

```python
class TestMyFeature:
    """Tests for my new feature."""

    def test_feature_works(self, model):
        """Test that my feature works correctly."""
        # Arrange
        input_data = {"name": "Test"}

        # Act
        result = model.predict(input_data)

        # Assert
        assert "classification" in result
        assert result["classification"] in ["Person", "Company", "University"]
```

### Integration Test Template

```python
class TestMyAPIEndpoint:
    """Tests for my new API endpoint."""

    def test_endpoint_success(self, api_client, model_name):
        """Test successful API call."""
        response = api_client.get(f"/v2/my-endpoint/{model_name}")

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
```

## Continuous Integration

To run tests in CI/CD:

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
python3 -m pytest src/tests/test_model.py -v --cov=src --cov-report=xml

# For API tests, start server first
docker-compose up -d
python3 -m pytest src/tests/test_api.py -v
docker-compose down
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Make sure you're in the project root
cd /path/to/melio

# Run tests with python module syntax
python3 -m pytest src/tests/test_model.py
```

### Coverage Not Working

Install pytest-cov:

```bash
pip install pytest-cov
```

### Tests Fail Randomly

The current classifier is random, so some tests verify randomness:

```python
def test_predict_is_random(self, model):
    """Test that the classifier produces varied results."""
    results = [model.predict({"name": "Test"})["classification"] for _ in range(20)]
    unique_results = set(results)
    assert len(unique_results) > 1  # Should see variation
```

This test verifies that the random classifier is working. Once you implement a real ML model, you can remove or modify this test.

## Test Organization

Tests are organized by functionality:

| Test Class | Purpose | Number of Tests |
|------------|---------|-----------------|
| `TestNameClassifierInitialization` | Model setup | 2 |
| `TestPreprocess` | Input processing | 5 |
| `TestPredict` | Prediction logic | 5 |
| `TestPostprocess` | Output formatting | 4 |
| `TestEndToEnd` | Full pipeline | 3 |

**Total:** 19 unit tests

## Next Steps

1. **Add more edge cases** - Test with unusual inputs
2. **Add performance tests** - Measure inference latency
3. **Add load tests** - Test under concurrent requests
4. **Mock external dependencies** - If you add external API calls
5. **Test with real data** - Use the provided CSV dataset

## Quick Reference

| Command | Description |
|---------|-------------|
| `pytest src/tests/test_model.py` | Run all unit tests |
| `pytest src/tests/test_model.py -v` | Verbose output |
| `pytest src/tests/test_model.py -k "preprocess"` | Run tests matching "preprocess" |
| `pytest src/tests/test_model.py --cov=src` | With coverage |
| `pytest src/tests/test_model.py -x` | Stop on first failure |
| `pytest src/tests/test_model.py --lf` | Run last failed tests |
| `pytest src/tests/ -v` | Run all tests (unit + integration) |

For more pytest options:
```bash
pytest --help
```
