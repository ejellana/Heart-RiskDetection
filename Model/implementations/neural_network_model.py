# Model/implementations/neural_network_model.py
import tensorflow as tf
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from ..interfaces.ml_model_interface import MLModelInterface

class NeuralNetworkModel(MLModelInterface):
    def __init__(self):
        self.model = None
        self.scaler = None
        model_dir = os.path.join(os.path.dirname(__file__), '../trained_models')
        try:
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'keras_dense_model.h5'))
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # Load the pre-trained preprocessor
            preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                print(f"Loaded pre-trained preprocessor from {preprocessor_path}")
            else:
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}. Please ensure preprocessor.pkl exists in the trained_models folder.")
        except Exception as e:
            raise RuntimeError(f"Error loading neural network model: {e}")

    def Predict(self, features):
        # Expected features based on training data
        expected_features = {
            'numerical': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'],
            'categorical': ['sex', 'cp', 'exang', 'slope', 'thal']
        }
        
        # Prepare input data as a DataFrame to match the preprocessor's structure
        import pandas as pd
        input_df = pd.DataFrame([features])
        
        # Transform input data using the pre-loaded preprocessor
        input_processed = self.preprocessor.transform(input_df)
        
        # Predict
        prediction_probs = self.model.predict(input_processed, verbose=0)
        cluster = int(np.argmax(prediction_probs, axis=1)[0])
        risk_level = {0: 'Low Risk (Cluster 1)', 1: 'Medium Risk (Cluster 2)', 2: 'High Risk (Cluster 3)'}[cluster]
        
        return {
            'cluster': cluster + 1,
            'risk_level': risk_level,
            'message': f'Patient is predicted to be in Cluster {cluster + 1} ({risk_level}).',
            'input': features,
            'model_type': 'Neural Network'
        }