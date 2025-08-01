# Model/implementations/heart_cluster_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
from ..interfaces.ml_model_interface import MLModelInterface

class HeartClusterModel(MLModelInterface):
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer_numeric = None
        self.imputer_categorical = None
        model_dir = os.path.join(os.path.dirname(__file__), '../trained_models')
        try:
            self.model = joblib.load(os.path.join(model_dir, 'heart_cluster_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            self.imputer_numeric = joblib.load(os.path.join(model_dir, 'imputer_numeric.pkl'))
            self.imputer_categorical = joblib.load(os.path.join(model_dir, 'imputer_categorical.pkl'))
        except Exception as e:
            raise RuntimeError(f"Error loading model files: {e}")

    def Predict(self, features):
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        categorical_cols = ['sex', 'cp', 'exang', 'slope', 'thal']
        # Create DataFrame with all expected columns, using defaults if not provided
        default_features = {
            'age': 0, 'sex': 0, 'cp': 0, 'trestbps': 0, 'chol': 0, 'thalach': 0,
            'exang': 0, 'oldpeak': 0, 'slope': 0, 'ca': 0, 'thal': 0
        }
        default_features.update(features)  # Override with provided features
        df = pd.DataFrame([default_features], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])
        
        # Impute and scale
        df[numeric_cols] = self.imputer_numeric.transform(df[numeric_cols])
        df[categorical_cols] = self.imputer_categorical.transform(df[categorical_cols]).astype(int)
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # Predict
        cluster = self.model.predict(df)[0]
        print(f"Predicted cluster: {cluster}, type: {type(cluster)}, features: {features}")  # Enhanced debug
        risk_map = {1: 'Low Risk', 2: 'Mid Risk', 3: 'High Risk'}
        return {
            'cluster': int(cluster),
            'risk_level': risk_map.get(cluster, 'Unknown Risk'),
            'message': f'Patient is predicted to be in Cluster {int(cluster)} ({risk_map.get(cluster, "Unknown Risk")}).',
            'input': {k: float(v) if isinstance(v, (int, np.integer, np.floating)) else v for k, v in features.items()},
            'model_type': 'Cluster'
        }