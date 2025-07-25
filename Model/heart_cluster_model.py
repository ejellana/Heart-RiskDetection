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
        
        # Convert 'Male'/'Female' to 0/1 for sex column
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).fillna(df['sex'])
        df['sex'] = self.imputer_categorical.fit_transform(df[['sex']]).ravel().astype(int)
        
        # Convert 'cluster1', 'cluster2', 'cluster3' to 1, 2, 3 with validation
        if df['cluster'].dtype == 'object':
            df['cluster'] = df['cluster'].str.extract('(\d+)').astype(int)
            if df['cluster'].isnull().any():
                raise ValueError("Failed to convert some cluster values to integers. Check dataset.")
        else:
            df['cluster'] = df['cluster'].astype(int)
        
        df[numeric_cols] = self.imputer_numeric.fit_transform(df[numeric_cols])
        df[categorical_cols] = self.imputer_categorical.fit_transform(df[categorical_cols])
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
        df = pd.DataFrame([features], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])
        df[numeric_cols] = self.imputer_numeric.transform(df[numeric_cols])
        df[categorical_cols] = self.imputer_categorical.transform(df[categorical_cols])
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        cluster = self.model.predict(df)[0]
        print(f"Predicted cluster: {cluster}, type: {type(cluster)}")  # Debug print
        risk_map = {1: 'Low Risk', 2: 'Mid Risk', 3: 'High Risk'}
        return {'cluster': int(cluster), 'risk_level': risk_map.get(cluster, 'Unknown Risk')}

# Train the model (run once)
if __name__ == "__main__":
    model = HeartClusterModel()
    try:
        df = model.load_and_preprocess('heartCluster.csv')
        model.train(df)
    except ValueError as e:
        print(f"Error during preprocessing: {e}")