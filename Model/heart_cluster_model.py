import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

class HeartClusterModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer_numeric = None
        self.imputer_categorical = None

    def load_and_preprocess(self, filepath):
        df = pd.read_csv(filepath)
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        categorical_cols = ['sex', 'cp', 'exang', 'slope', 'thal']
        
        # Ensure 'sex' is numeric (0/1)
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).fillna(df['sex']).astype(float)
        df['sex'] = self.imputer_categorical.fit_transform(df[['sex']]).ravel().astype(int)
        
        # Convert 'cluster1', 'cluster2', 'cluster3' to 1, 2, 3
        if df['cluster'].dtype == 'object':
            df['cluster'] = df['cluster'].str.extract('(\d+)').astype(int)
            if df['cluster'].isnull().any():
                raise ValueError("Failed to convert some cluster values to integers. Check dataset.")
        else:
            df['cluster'] = df['cluster'].astype(int)
        
        # Impute and scale
        df[numeric_cols] = self.imputer_numeric.fit_transform(df[numeric_cols])
        df[categorical_cols] = self.imputer_categorical.fit_transform(df[categorical_cols]).astype(int)
        return df

    def train(self, df):
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
        y = df['cluster']
        self.scaler = StandardScaler()
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        joblib.dump(self.model, 'heart_cluster_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.imputer_numeric, 'imputer_numeric.pkl')
        joblib.dump(self.imputer_categorical, 'imputer_categorical.pkl')

    def predict(self, features):
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
        return {'cluster': int(cluster), 'risk_level': risk_map.get(cluster, 'Unknown Risk')}

if __name__ == "__main__":
    model = HeartClusterModel()
    try:
        df = model.load_and_preprocess('heartCluster.csv')
        model.train(df)
    except ValueError as e:
        print(f"Error during preprocessing: {e}")