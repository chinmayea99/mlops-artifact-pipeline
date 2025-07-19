import json
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_model(model_path):
    """Load trained model from pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_digits_data():
    """Load and return digits dataset."""
    digits = load_digits()
    return digits.data, digits.target 
