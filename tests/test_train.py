import pytest
import json
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import load_config, train_model

class TestTrainingPipeline:
    
    def test_config_loading(self):
        """Test that configuration file loads successfully."""
        config = load_config('config/config.json')
        assert isinstance(config, dict)
        assert 'model_params' in config
    
    def test_config_parameters(self):
        """Test that all required hyperparameters exist with correct types."""
        config = load_config('config/config.json')
        model_params = config['model_params']
        
        # Check required parameters exist
        assert 'C' in model_params
        assert 'solver' in model_params
        assert 'max_iter' in model_params
        
        # Check data types
        assert isinstance(model_params['C'], (int, float))
        assert isinstance(model_params['solver'], str)
        assert isinstance(model_params['max_iter'], int)
    
    def test_model_creation(self):
        """Test that training function returns a LogisticRegression object."""
        config = load_config('config/config.json')
        digits = load_digits()
        X, y = digits.data, digits.target
        
        model = train_model(X, y, config)
        
        # Verify model type
        assert isinstance(model, LogisticRegression)
        
        # Verify model is fitted (check for fitted attributes)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'classes_')
    
    def test_model_accuracy(self):
        """Test that model achieves reasonable accuracy."""
        config = load_config('config/config.json')
        digits = load_digits()
        X, y = digits.data, digits.target
        
        model = train_model(X, y, config)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Should achieve at least 90% accuracy on training data
        assert accuracy > 0.9, f"Accuracy {accuracy} is below threshold" 
