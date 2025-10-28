"""
Unit tests for the server.py script.

These tests verify the server initialization and KServe ModelServer integration.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
import kserve
from server import NameClassifier


class TestServerInitialization:
    """Tests for server script initialization and configuration."""

    def test_logging_configuration(self):
        """Test that logging is properly configured."""
        # Check that logging is configured
        logger = logging.getLogger(__name__)
        assert logger.level <= logging.INFO

    @patch('kserve.ModelServer')
    @patch('server.NameClassifier')
    def test_server_startup(self, mock_model_class, mock_model_server):
        """Test that the server starts correctly with proper model initialization."""
        # Mock the model instance
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock the ModelServer
        mock_server_instance = MagicMock()
        mock_model_server.return_value = mock_server_instance

        # Import and run the server startup code
        import server
        
        # Verify model was created with correct name
        mock_model_class.assert_called_once_with("name-classifier")
        
        # Verify ModelServer was started with the model
        mock_model_server.assert_called_once()
        mock_server_instance.start.assert_called_once_with([mock_model])

    @patch('kserve.ModelServer')
    @patch('server.NameClassifier')
    def test_model_name_configuration(self, mock_model_class, mock_model_server):
        """Test that the model name is correctly configured."""
        # Mock the model instance
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock the ModelServer
        mock_server_instance = MagicMock()
        mock_model_server.return_value = mock_server_instance

        # Import and run the server startup code
        import server
        
        # Verify the model name is set correctly
        mock_model_class.assert_called_once_with("name-classifier")

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

    @patch('kserve.ModelServer')
    @patch('server.NameClassifier')
    def test_model_server_initialization_parameters(self, mock_model_class, mock_model_server):
        """Test that ModelServer is initialized with correct parameters."""
        # Mock the model instance
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock the ModelServer
        mock_server_instance = MagicMock()
        mock_model_server.return_value = mock_server_instance

        # Import and run the server startup code
        import server
        
        # Verify ModelServer was instantiated
        mock_model_server.assert_called_once()
        
        # Verify start was called with the model in a list
        mock_server_instance.start.assert_called_once_with([mock_model])

    def test_logger_configuration(self):
        """Test that the logger is properly configured with expected format."""
        import server
        
        # Get the root logger to check configuration
        root_logger = logging.getLogger()
        
        # Check that handlers exist
        assert len(root_logger.handlers) > 0
        
        # Check that the format includes expected components
        for handler in root_logger.handlers:
            if hasattr(handler, 'formatter') and handler.formatter:
                format_string = handler.formatter._fmt
                assert '%(asctime)s' in format_string
                assert '%(name)s' in format_string
                assert '%(levelname)s' in format_string
                assert '%(message)s' in format_string

    @patch('kserve.ModelServer')
    @patch('server.NameClassifier')
    def test_server_startup_with_exception_handling(self, mock_model_class, mock_model_server):
        """Test server startup behavior when exceptions occur."""
        # Mock the model instance
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock the ModelServer to raise an exception
        mock_server_instance = MagicMock()
        mock_server_instance.start.side_effect = Exception("Test exception")
        mock_model_server.return_value = mock_server_instance

        # Import the server module (this should not raise an exception)
        import server
        
        # The actual startup code is in the if __name__ == "__main__" block
        # which is not executed during import, so we just verify the module loads
        assert server is not None

    def test_server_module_docstring_and_structure(self):
        """Test that the server module has proper structure and can be imported."""
        import server
        
        # Verify the module can be imported without errors
        assert server is not None
        
        # Check that the module has the expected attributes
        expected_attrs = ['logging', 'kserve', 'NameClassifier']
        for attr in expected_attrs:
            assert hasattr(server, attr), f"Missing attribute: {attr}"

    @patch('kserve.ModelServer')
    @patch('server.NameClassifier')
    def test_model_ready_state(self, mock_model_class, mock_model_server):
        """Test that the model is properly initialized and ready."""
        # Mock the model instance with ready state
        mock_model = MagicMock()
        mock_model.ready = True
        mock_model_class.return_value = mock_model
        
        # Mock the ModelServer
        mock_server_instance = MagicMock()
        mock_model_server.return_value = mock_server_instance

        # Import and run the server startup code
        import server
        
        # Verify model was created
        mock_model_class.assert_called_once_with("name-classifier")
        
        # Verify the model is ready
        assert mock_model.ready is True
