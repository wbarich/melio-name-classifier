"""
Integration tests for the KServe API endpoints.

These tests verify the HTTP API endpoints conform to the V2 protocol.
"""

import pytest
import httpx


# Base URL for the API (assumes server is running locally)
BASE_URL = "http://localhost:8080"


@pytest.fixture
def api_client():
    """
    Fixture providing an HTTP client for API testing.

    Returns:
        httpx.Client: Synchronous HTTP client
    """
    return httpx.Client(base_url=BASE_URL, timeout=10.0)


@pytest.fixture
def model_name():
    """
    Fixture providing the model name.

    Returns:
        str: Model name
    """
    return "name-classifier"


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_live_endpoint(self, api_client):
        """Test the /v2/health/live endpoint."""
        response = api_client.get("/v2/health/live")

        assert response.status_code == 200
        data = response.json()
        assert "live" in data
        assert data["live"] is True

    def test_health_ready_endpoint(self, api_client):
        """Test the /v2/health/ready endpoint."""
        response = api_client.get("/v2/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert data["ready"] is True


class TestModelMetadata:
    """Tests for model metadata endpoints."""

    def test_model_metadata_endpoint(self, api_client, model_name):
        """Test the /v2/models/{model_name} endpoint."""
        response = api_client.get(f"/v2/models/{model_name}")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == model_name

    def test_model_metadata_structure(self, api_client, model_name):
        """Test that model metadata has the expected structure."""
        response = api_client.get(f"/v2/models/{model_name}")
        data = response.json()

        # V2 protocol expected fields
        assert "name" in data
        assert "platform" in data
        assert "inputs" in data
        assert "outputs" in data


class TestInferenceEndpoint:
    """Tests for the inference endpoint."""

    @pytest.fixture
    def valid_inference_request(self):
        """
        Fixture providing a valid V2 inference request.

        Returns:
            dict: Valid inference request payload
        """
        return {
            "inputs": [
                {
                    "name": "name",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Bob Immerman"]
                }
            ]
        }

    def test_inference_endpoint_success(self, api_client, model_name, valid_inference_request):
        """Test successful inference request."""
        response = api_client.post(
            f"/v2/models/{model_name}/infer",
            json=valid_inference_request
        )

        assert response.status_code == 200

    def test_inference_response_structure(self, api_client, model_name, valid_inference_request):
        """Test that inference response has correct V2 structure."""
        response = api_client.post(
            f"/v2/models/{model_name}/infer",
            json=valid_inference_request
        )

        data = response.json()

        # Check required V2 response fields
        assert "model_name" in data
        assert "outputs" in data
        assert data["model_name"] == model_name

        # Check outputs structure
        assert len(data["outputs"]) == 1
        output = data["outputs"][0]
        assert "name" in output
        assert "shape" in output
        assert "datatype" in output
        assert "data" in output

    def test_inference_returns_valid_classification(self, api_client, model_name, valid_inference_request):
        """Test that inference returns a valid classification."""
        response = api_client.post(
            f"/v2/models/{model_name}/infer",
            json=valid_inference_request
        )

        data = response.json()
        classification = data["outputs"][0]["data"][0]

        assert classification in ["Person", "Company", "University"]

    def test_inference_with_person_name(self, api_client, model_name):
        """Test inference with a person name."""
        request = {
            "inputs": [
                {
                    "name": "name",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Dr. Jane Smith"]
                }
            ]
        }

        response = api_client.post(f"/v2/models/{model_name}/infer", json=request)

        assert response.status_code == 200
        data = response.json()
        classification = data["outputs"][0]["data"][0]
        assert classification in ["Person", "Company", "University"]

    def test_inference_with_company_name(self, api_client, model_name):
        """Test inference with a company name."""
        request = {
            "inputs": [
                {
                    "name": "name",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Microsoft Corporation"]
                }
            ]
        }

        response = api_client.post(f"/v2/models/{model_name}/infer", json=request)

        assert response.status_code == 200
        data = response.json()
        classification = data["outputs"][0]["data"][0]
        assert classification in ["Person", "Company", "University"]

    def test_inference_with_university_name(self, api_client, model_name):
        """Test inference with a university name."""
        request = {
            "inputs": [
                {
                    "name": "name",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Harvard University"]
                }
            ]
        }

        response = api_client.post(f"/v2/models/{model_name}/infer", json=request)

        assert response.status_code == 200
        data = response.json()
        classification = data["outputs"][0]["data"][0]
        assert classification in ["Person", "Company", "University"]

    def test_inference_with_multiple_requests(self, api_client, model_name):
        """Test making multiple inference requests in sequence."""
        test_names = [
            "Bob Immerman",
            "Microsoft Corporation",
            "Stanford University",
            "Dr. Alice Brown",
            "Apple Inc."
        ]

        for name in test_names:
            request = {
                "inputs": [
                    {
                        "name": "name",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [name]
                    }
                ]
            }

            response = api_client.post(f"/v2/models/{model_name}/infer", json=request)

            assert response.status_code == 200
            data = response.json()
            classification = data["outputs"][0]["data"][0]
            assert classification in ["Person", "Company", "University"]

    def test_inference_with_special_characters(self, api_client, model_name):
        """Test inference with names containing special characters."""
        special_names = [
            "Jean-Claude Van Damme",
            "O'Brien",
            "Université D'Artois",
            "Company & Associates LLC"
        ]

        for name in special_names:
            request = {
                "inputs": [
                    {
                        "name": "name",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [name]
                    }
                ]
            }

            response = api_client.post(f"/v2/models/{model_name}/infer", json=request)
            assert response.status_code == 200
            data = response.json()
            assert data["outputs"][0]["data"][0] in ["Person", "Company", "University"]


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_model_name(self, api_client):
        """Test requesting a non-existent model."""
        response = api_client.get("/v2/models/nonexistent-model")

        # Should return 404 or similar error
        assert response.status_code in [404, 500]

    def test_inference_empty_payload(self, api_client, model_name):
        """Test inference with empty request payload."""
        response = api_client.post(
            f"/v2/models/{model_name}/infer",
            json={}
        )

        # Should return error (400 or 500)
        assert response.status_code in [400, 422, 500]

    def test_inference_missing_inputs(self, api_client, model_name):
        """Test inference with missing inputs field."""
        request = {
            "invalid_field": "data"
        }

        response = api_client.post(
            f"/v2/models/{model_name}/infer",
            json=request
        )

        # Should return validation error
        assert response.status_code in [400, 422, 500]

    def test_inference_invalid_content_type(self, api_client, model_name):
        """Test inference with invalid content type."""
        response = api_client.post(
            f"/v2/models/{model_name}/infer",
            content="invalid data",
            headers={"Content-Type": "text/plain"}
        )

        # Should return error
        assert response.status_code in [400, 415, 422, 500]


class TestV2ProtocolCompliance:
    """Tests for KServe V2 protocol compliance."""

    def test_response_content_type(self, api_client, model_name):
        """Test that responses have correct content type."""
        # Test health endpoint
        response = api_client.get("/v2/health/live")
        assert "application/json" in response.headers.get("content-type", "")

        # Test inference endpoint
        request = {
            "inputs": [
                {
                    "name": "name",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Test"]
                }
            ]
        }
        response = api_client.post(f"/v2/models/{model_name}/infer", json=request)
        assert "application/json" in response.headers.get("content-type", "")

    def test_inference_output_datatype(self, api_client, model_name):
        """Test that output datatype is correct."""
        request = {
            "inputs": [
                {
                    "name": "name",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": ["Test Name"]
                }
            ]
        }

        response = api_client.post(f"/v2/models/{model_name}/infer", json=request)
        data = response.json()

        output = data["outputs"][0]
        assert output["datatype"] == "BYTES"
        assert output["shape"] == [1]
        assert len(output["data"]) == 1
