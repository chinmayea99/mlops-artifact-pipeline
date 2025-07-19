import pickle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_model(model_path):
    """Load trained model from pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def main():
    try:
        # Load the trained model
        model = load_model('models/model_train.pkl')
        print("Model loaded successfully!")
        
        # Load dataset for inference
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        print(f"\nInference Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Show sample predictions
        print(f"\nSample Predictions (first 10):")
        for i in range(10):
            print(f"True: {y[i]}, Predicted: {y_pred[i]}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y, y_pred))
        
    except FileNotFoundError:
        print("Error: Model file 'models/model_train.pkl' not found!")
        print("Please run training first or download the model artifact.")
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main() 
