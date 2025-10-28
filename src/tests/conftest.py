"""
Pytest configuration and fixtures for the Name Classifier tests.

This module provides reusable fixtures for testing the KServe model.
"""

import pytest
from kserve import InferRequest, InferInput
from model import NameClassifier


@pytest.fixture
def model():
    """
    Fixture that provides a NameClassifier instance.

    Returns:
        NameClassifier: An initialized name classifier model
    """
    return NameClassifier(name="name-classifier")


@pytest.fixture
def sample_person_names():
    """
    Fixture providing sample person names for testing.

    Returns:
        list[str]: List of person names
    """
    return [
        "Bob Immerman",
        "Dr. Jane Smith",
        "Prof. John Doe",
        "Ms. Sarah Johnson",
        "Mr. Michael Chen"
    ]


@pytest.fixture
def sample_company_names():
    """
    Fixture providing sample company names for testing.

    Returns:
        list[str]: List of company names
    """
    return [
        "Microsoft Corporation",
        "Apple Inc.",
        "ACME Industries Ltd.",
        "Google LLC",
        "Amazon.com, Inc."
    ]


@pytest.fixture
def sample_university_names():
    """
    Fixture providing sample university names for testing.

    Returns:
        list[str]: List of university names
    """
    return [
        "Harvard University",
        "Stanford University",
        "Massachusetts Institute of Technology",
        "Université de Paris",
        "Oxford University"
    ]


@pytest.fixture
def all_sample_names(sample_person_names, sample_company_names, sample_university_names):
    """
    Fixture combining all sample names into a single dict.

    Args:
        sample_person_names: Person names fixture
        sample_company_names: Company names fixture
        sample_university_names: University names fixture

    Returns:
        dict: Dictionary with 'person', 'company', and 'university' keys
    """
    return {
        "person": sample_person_names,
        "company": sample_company_names,
        "university": sample_university_names
    }


@pytest.fixture
def valid_categories():
    """
    Fixture providing the valid classification categories.

    Returns:
        list[str]: Valid category names
    """
    return ["Person", "Company", "University"]


def create_infer_request(name: str) -> InferRequest:
    """
    Helper function to create a KServe InferRequest object.

    Args:
        name: The name string to classify

    Returns:
        InferRequest: A properly formatted inference request
    """
    infer_input = InferInput(
        name="name",
        shape=[1],
        datatype="BYTES",
        data=[name]
    )
    return InferRequest(
        model_name="name-classifier",
        infer_inputs=[infer_input]
    )


@pytest.fixture
def infer_request_factory():
    """
    Fixture providing a factory function to create InferRequest objects.

    Returns:
        callable: Factory function that takes a name string and returns InferRequest
    """
    return create_infer_request


@pytest.fixture
def sample_infer_request():
    """
    Fixture providing a sample InferRequest for testing.

    Returns:
        InferRequest: Request with "Bob Immerman" as test data
    """
    return create_infer_request("Bob Immerman")


@pytest.fixture
def edge_case_names():
    """
    Fixture providing edge case names for testing.

    Returns:
        list[str]: Names with special characters, multiple words, etc.
    """
    return [
        "",  # Empty string
        "A",  # Single character
        "Jean-Claude Van Damme",  # Hyphenated name
        "O'Brien",  # Apostrophe
        "Université D'Artois",  # Accents and apostrophe
        "ALL CAPS NAME",  # All uppercase
        "lowercase name",  # All lowercase
        "Name123",  # Contains numbers
        "Dr. Prof. Sir John Smith III",  # Multiple titles
        "Company & Associates LLC",  # Special characters
    ]


@pytest.fixture
def mock_prediction_data():
    """
    Fixture providing mock data for the predict method.

    Returns:
        dict: Sample preprocessed data
    """
    return {"name": "Bob Immerman"}


@pytest.fixture
def expected_v2_response_structure():
    """
    Fixture defining the expected structure of a V2 inference response.

    Returns:
        dict: Expected keys in the response
    """
    return {
        "required_keys": ["model_name", "outputs"],
        "output_keys": ["name", "shape", "datatype", "data"]
    }
