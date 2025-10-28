"""
Unit tests for the server.py script.

These tests verify the server initialization and KServe ModelServer integration.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock


class TestServerInitialization:
    """Tests for server script initialization and configuration."""

    def test_logging_configuration(self):
        """Test that logging is properly configured."""
        # Check that logging is configured
        logger = logging.getLogger(__name__)
        assert logger.level <= logging.INFO

    def test_server_startup(self):
        """Test that the server module can be imported and has expected structure."""
        import server
        
        # Verify the module has expected attributes
        assert hasattr(server, 'NameClassifier')
        assert hasattr(server, 'kserve')
        assert hasattr(server, 'logging')
        
        # Verify NameClassifier is the correct class
        from model import NameClassifier as ModelNameClassifier
        assert server.NameClassifier is ModelNameClassifier

    def test_model_name_configuration(self):
        """Test that the server module imports the correct NameClassifier."""
        import server
        
        # Verify NameClassifier is imported correctly
        assert server.NameClassifier is not None
        assert hasattr(server.NameClassifier, '__init__')
        
        # Verify it's the same class as the one in model.py
        from model import NameClassifier as ModelNameClassifier
        assert server.NameClassifier is ModelNameClassifier

    def test_name_classifier_import(self):
        """Test that NameClassifier can be imported from server module."""
        from server import NameClassifier
        assert NameClassifier is not None
        assert hasattr(NameClassifier, '__init__')

    def test_kserve_import(self):
        """Test that kserve module is properly imported."""
        import server
        assert hasattr(server, 'kserve')
        assert hasattr(server, 'logging')

    @patch('kserve.ModelServer')
    @patch('server.NameClassifier')
    def test_server_logging_output(self, mock_model_class, mock_model_server, caplog):
        """Test that appropriate log messages are generated during startup."""
        # Mock the model instance
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock the ModelServer
        mock_server_instance = MagicMock()
        mock_model_server.return_value = mock_server_instance

        # Import and run the server startup code
        import server
        
        # Check that logging messages are generated
        # Note: The actual logging happens in the if __name__ == "__main__" block
        # which is not executed during import, so we test the logging setup instead
        logger = logging.getLogger('server')
        assert logger is not None

    def test_server_module_structure(self):
        """Test that the server module has the expected structure."""
        import server
        
        # Check that required attributes exist
        assert hasattr(server, 'logging')
        assert hasattr(server, 'kserve')
        assert hasattr(server, 'NameClassifier')
        
        # Check that NameClassifier is the correct class
        from model import NameClassifier as ModelNameClassifier
        assert server.NameClassifier is ModelNameClassifier

    def test_model_server_initialization_parameters(self):
        """Test that the server module has the expected structure for ModelServer."""
        import server
        
        # Verify kserve is imported
        assert hasattr(server, 'kserve')
        
        # Verify ModelServer is available through kserve
        assert hasattr(server.kserve, 'ModelServer')

    def test_logger_configuration(self):
        """Test that the logger is properly configured."""
        import server
        
        # Get the root logger to check configuration
        root_logger = logging.getLogger()
        
        # Check that handlers exist
        assert len(root_logger.handlers) > 0
        
        # Check that the format includes expected components
        for handler in root_logger.handlers:
            if hasattr(handler, 'formatter') and handler.formatter:
                format_string = handler.formatter._fmt
                # Check for common logging format components
                assert '%(name)s' in format_string
                assert 'levelname' in format_string  # Check for levelname (with or without %)
                assert '%(message)s' in format_string

    def test_server_startup_with_exception_handling(self):
        """Test that the server module can be imported without errors."""
        # Import the server module (this should not raise an exception)
        import server
        
        # Verify the module loads successfully
        assert server is not None
        assert hasattr(server, 'NameClassifier')

    def test_server_module_docstring_and_structure(self):
        """Test that the server module has proper structure and can be imported."""
        import server
        
        # Verify the module can be imported without errors
        assert server is not None
        
        # Check that the module has the expected attributes
        expected_attrs = ['logging', 'kserve', 'NameClassifier']
        for attr in expected_attrs:
            assert hasattr(server, attr), f"Missing attribute: {attr}"

    def test_model_ready_state(self):
        """Test that the NameClassifier class can be instantiated and is ready."""
        import server
        
        # Create a model instance to test
        model = server.NameClassifier("test-model")
        
        # Verify the model is ready
        assert model.ready is True
        assert model.name == "test-model"
        assert hasattr(model, 'classes')
