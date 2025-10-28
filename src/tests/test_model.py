"""
Unit tests for the NameClassifier model.

These tests verify the behavior of the preprocess, predict, and postprocess methods.
"""

import pytest
from kserve import InferRequest, InferResponse
from model import NameClassifier


class TestNameClassifierInitialization:
    """Tests for model initialization."""

    def test_model_initialization(self, model):
        """Test that the model initializes correctly."""
        assert model.name == "name-classifier"
        assert model.ready is True
        assert model.classes == ["Person", "Company", "University"]

    def test_model_has_required_methods(self, model):
        """Test that the model has all required methods."""
        assert hasattr(model, 'preprocess')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'postprocess')
        assert callable(model.preprocess)
        assert callable(model.predict)
        assert callable(model.postprocess)


class TestPreprocess:
    """Tests for the preprocess method."""

    def test_preprocess_valid_request(self, model, sample_infer_request):
        """Test preprocessing a valid inference request."""
        result = model.preprocess(sample_infer_request)

        assert isinstance(result, dict)
        assert "name" in result
        assert result["name"] == "Bob Immerman"

    def test_preprocess_extracts_name(self, model, infer_request_factory):
        """Test that preprocess correctly extracts the name from the request."""
        test_name = "Test Name"
        request = infer_request_factory(test_name)
        result = model.preprocess(request)

        assert result["name"] == test_name

    def test_preprocess_multiple_names(self, model, infer_request_factory, sample_person_names):
        """Test preprocessing multiple different names."""
        for name in sample_person_names:
            request = infer_request_factory(name)
            result = model.preprocess(request)
            assert result["name"] == name

    def test_preprocess_empty_inputs_raises_error(self, model):
        """Test that preprocess raises an error when inputs are empty."""
        # Create request with no inputs
        request = InferRequest(model_name="name-classifier", infer_inputs=[])

        with pytest.raises(ValueError, match="No inputs provided"):
            model.preprocess(request)

    def test_preprocess_empty_data_raises_error(self, model):
        """Test that preprocess raises an error when data is empty."""
        from kserve import InferInput

        # Create input with empty data
        infer_input = InferInput(
            name="name",
            shape=[1],
            datatype="BYTES",
            data=[]  # Empty data
        )
        request = InferRequest(model_name="name-classifier", infer_inputs=[infer_input])

        with pytest.raises(ValueError, match="No data provided"):
            model.preprocess(request)


class TestPredict:
    """Tests for the predict method."""

    def test_predict_returns_dict(self, model, mock_prediction_data):
        """Test that predict returns a dictionary."""
        result = model.predict(mock_prediction_data)

        assert isinstance(result, dict)
        assert "classification" in result

    def test_predict_returns_valid_category(self, model, mock_prediction_data, valid_categories):
        """Test that predict returns one of the valid categories."""
        result = model.predict(mock_prediction_data)
        classification = result["classification"]

        assert classification in valid_categories

    def test_predict_with_different_names(self, model, all_sample_names):
        """Test prediction with various name types."""
        for category, names in all_sample_names.items():
            for name in names:
                data = {"name": name}
                result = model.predict(data)

                assert "classification" in result
                assert result["classification"] in ["Person", "Company", "University"]

    def test_predict_is_random(self, model):
        """Test that the current naive classifier produces varied results."""
        # Run prediction multiple times with the same input
        name = "Test Name"
        data = {"name": name}
        results = [model.predict(data)["classification"] for _ in range(20)]

        # With random selection, we should see some variation in 20 tries
        # (There's a very small chance this could fail with true randomness)
        unique_results = set(results)
        assert len(unique_results) > 1, "Random classifier should produce varied results"

    def test_predict_with_edge_cases(self, model, edge_case_names):
        """Test prediction with edge case names."""
        for name in edge_case_names:
            data = {"name": name}
            result = model.predict(data)

            assert "classification" in result
            assert result["classification"] in ["Person", "Company", "University"]


class TestPostprocess:
    """Tests for the postprocess method."""

    def test_postprocess_returns_infer_response(self, model):
        """Test that postprocess returns an InferResponse object."""
        result_data = {"classification": "Person"}
        response = model.postprocess(result_data)

        assert isinstance(response, InferResponse)

    def test_postprocess_has_correct_structure(self, model):
        """Test that the response has the correct V2 protocol structure."""
        result_data = {"classification": "Company"}
        response = model.postprocess(result_data)

        # Check model name
        assert response.model_name == "name-classifier"

        # Check outputs exist
        assert response.outputs is not None
        assert len(response.outputs) == 1

    def test_postprocess_output_contains_classification(self, model):
        """Test that the output contains the classification result."""
        classification = "University"
        result_data = {"classification": classification}
        response = model.postprocess(result_data)

        output = response.outputs[0]
        assert output.name == "classification"
        assert output.shape == [1]
        assert output.datatype == "BYTES"
        assert output.data == [classification]

    def test_postprocess_all_categories(self, model, valid_categories):
        """Test postprocessing with all valid categories."""
        for category in valid_categories:
            result_data = {"classification": category}
            response = model.postprocess(result_data)

            output = response.outputs[0]
            assert output.data == [category]


class TestEndToEnd:
    """End-to-end integration tests for the full pipeline."""

    def test_full_pipeline(self, model, infer_request_factory, sample_person_names):
        """Test the complete preprocess → predict → postprocess pipeline."""
        for name in sample_person_names:
            # Create request
            request = infer_request_factory(name)

            # Preprocess
            preprocessed = model.preprocess(request)
            assert preprocessed["name"] == name

            # Predict
            prediction = model.predict(preprocessed)
            assert "classification" in prediction
            assert prediction["classification"] in ["Person", "Company", "University"]

            # Postprocess
            response = model.postprocess(prediction)
            assert isinstance(response, InferResponse)
            assert response.model_name == "name-classifier"
            assert len(response.outputs) == 1
            assert response.outputs[0].data[0] in ["Person", "Company", "University"]

    def test_pipeline_preserves_data_flow(self, model, infer_request_factory):
        """Test that data flows correctly through the entire pipeline."""
        test_name = "Integration Test Name"
        request = infer_request_factory(test_name)

        # Run through pipeline
        preprocessed = model.preprocess(request)
        prediction = model.predict(preprocessed)
        response = model.postprocess(prediction)

        # Verify the name was processed (we can't verify exact classification due to randomness)
        # But we can verify the structure is correct
        assert response.outputs[0].name == "classification"
        assert len(response.outputs[0].data) == 1
        assert isinstance(response.outputs[0].data[0], str)

    def test_pipeline_handles_multiple_requests(self, model, infer_request_factory, all_sample_names):
        """Test processing multiple requests in sequence."""
        all_names = []
        for names_list in all_sample_names.values():
            all_names.extend(names_list)

        results = []
        for name in all_names:
            request = infer_request_factory(name)
            preprocessed = model.preprocess(request)
            prediction = model.predict(preprocessed)
            response = model.postprocess(prediction)
            results.append(response.outputs[0].data[0])

        # All results should be valid categories
        assert all(r in ["Person", "Company", "University"] for r in results)
        # We should have results for all test names
        assert len(results) == len(all_names)
