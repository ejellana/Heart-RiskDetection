# Controller/app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, abort
from Model.heart_cluster_model import HeartClusterModel
import joblib
import os
import json
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, template_folder='../view', static_folder='../static')
app.secret_key = os.urandom(24)  # Secure random key

# Load pre-trained models and preprocessors
model_dir = os.path.join(os.path.dirname(__file__), '..')  # Project root
try:
    cluster_model = HeartClusterModel()
    cluster_model.model = joblib.load(os.path.join(model_dir, 'heart_cluster_model.pkl'))
    cluster_model.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    cluster_model.imputer_numeric = joblib.load(os.path.join(model_dir, 'imputer_numeric.pkl'))
    cluster_model.imputer_categorical = joblib.load(os.path.join(model_dir, 'imputer_categorical.pkl'))
except Exception as e:
    print(f"Error loading clustering model files: {e}")
    raise

try:
    keras_model = tf.keras.models.load_model(os.path.join(model_dir, 'keras_dense_model.h5'))
    keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Suppress metrics warning
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
except Exception as e:
    print(f"Error loading Keras model or preprocessor: {e}")
    raise

# Feature lists
CLUSTER_FEATURES = ['age', 'sex', 'cp', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
KERAS_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'sex', 'cp', 'exang', 'slope', 'thal']

# MySQL Configuration
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Adjust for your MySQL setup
        database="heartPrediction"
    )
    cursor = db.cursor()
except mysql.connector.Error as e:
    print(f"Database connection error: {e}")
    raise

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    print(f"Request method: {request.method}")
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT id, username, name, password, specialty FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        print(f"User query result: {user}")
        if user and check_password_hash(user[3], password):
            session['username'] = user[1]
            session['id'] = user[0]
            session['name'] = user[2]
            session['specialty'] = user[4]
            print(f"Session name set to: {session['name']}, specialty: {session['specialty']}")
            return redirect(url_for('model_selection'))
        else:
            return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('id', None)
    session.pop('name', None)  # Clear the name from the session
    session.pop('specialty', None)  # Clear the specialty from the session
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        specialty = request.form['specialty']
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return render_template('register.html', error="Username already taken.")
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        sql = "INSERT INTO users (name, username, password, specialty) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (name, username, hashed_password, specialty))
        db.commit()
        return redirect(url_for('login') + '?success=true')
    return render_template('register.html')

@app.route('/model_selection', methods=['GET', 'POST'])
def model_selection():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        model_type = request.form['model_type']
        if model_type == 'cluster':
            return redirect(url_for('predict'))
        elif model_type == 'neural':
            return redirect(url_for('neural_input'))
    return render_template('model_selection.html', name=session.get('name', 'Guest'), specialty=session.get('specialty', 'Specialist'))

@app.route('/predict')
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/neural_input')
def neural_input():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('neural_input.html', name=session.get('name', 'Guest'), specialty=session.get('specialty', 'Specialist'))

@app.route('/submit', methods=['POST'])
def submit():
    if 'username' not in session:
        return redirect(url_for('login'))
    try:
        data = request.form
        model_type = data['model_type']
        response = {}
        
        if model_type == 'cluster':
            required_fields = ['age', 'sex', 'cp', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
            try:
                features = {
                    'age': float(data['age']),
                    'sex': int(data['sex']),
                    'cp': int(data['cp']),
                    'trestbps': float(data.get('trestbps', cluster_model.imputer_numeric.statistics_[1])),
                    'chol': float(data['chol']),
                    'thalach': float(data['thalach']),
                    'exang': int(data.get('exang', cluster_model.imputer_categorical.statistics_[2])),
                    'oldpeak': float(data['oldpeak']),
                    'slope': int(data.get('slope', cluster_model.imputer_categorical.statistics_[3])),
                    'ca': int(data['ca']),
                    'thal': int(data['thal'])
                }
                result = cluster_model.predict(features)
                # Convert NumPy types to native Python types
                response = {
                    'cluster': int(result['cluster']),  # Ensure integer
                    'risk_level': result['risk_level'],
                    'message': f'Patient is predicted to be in Cluster {int(result["cluster"])} ({result["risk_level"]}).',
                    'input': {k: float(v) if isinstance(v, (int, np.integer, np.floating)) else v for k, v in features.items()},
                    'model_type': 'Cluster'
                }
            except ValueError as ve:
                return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
        else:
            features = {
                'age': float(data['age']),
                'trestbps': float(data['trestbps']),
                'chol': float(data['chol']),
                'thalach': float(data['thalach']),
                'oldpeak': float(data['oldpeak']),
                'ca': int(data['ca']),
                'sex': int(data['sex']),
                'cp': int(data['cp']),
                'exang': int(data['exang']),
                'slope': int(data['slope']),
                'thal': int(data['thal'])
            }
            input_data = pd.DataFrame([features], columns=KERAS_FEATURES)
            input_processed = preprocessor.transform(input_data)
            prediction_probs = keras_model.predict(input_processed, verbose=0)
            cluster = int(np.argmax(prediction_probs, axis=1)[0])
            risk_level = {0: 'Low Risk (Cluster 1)', 1: 'Medium Risk (Cluster 2)', 2: 'High Risk (Cluster 3)'}[cluster]
            response = {
                'cluster': cluster + 1,
                'risk_level': risk_level,
                'message': f'Patient is predicted to be in Cluster {cluster + 1} ({risk_level}).',
                'input': features,
                'model_type': 'Neural Network'
            }
            print(f"Neural Network response: {response}")  # Debug
        
        session['result'] = response
        print(f"Response set in session: {response}")
        # Store prediction in database
        user_id = session['id']
        prediction_data = json.dumps(response)
        cursor.execute(
            "INSERT INTO predictions (user_id, prediction_data, model_type, timestamp) VALUES (%s, %s, %s, NOW())",
            (user_id, prediction_data, response['model_type'])
        )
        db.commit()

        print(f"Response JSON: {json.dumps(response)}")
        return redirect(url_for('result'))
    except ValueError as ve:
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/result')
def result():
    if 'username' not in session:
        return redirect(url_for('login'))
    result = session.get('result')
    if not result:
        print("Warning: No result data in session")
        return redirect(url_for('model_selection'))
    session.pop('result', None)
    print(f"Result from session: {result}")
    return render_template('results.html', result=result, model_type=result.get('model_type', 'Cluster'))

@app.route('/records')
def records():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_id = session['id']
    session.pop('records', None)
    cursor.execute("SELECT id, prediction_data, model_type FROM predictions WHERE user_id = %s ORDER BY timestamp DESC", (user_id,))
    records = [{'id': row[0], 'data': json.loads(row[1]), 'model_type': row[2]} for row in cursor.fetchall()]
    print(f"Fetched records count: {len(records)}")
    return render_template('records.html', records=records)

@app.route('/delete_record/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    user_id = session['id']
    try:
        cursor.execute("DELETE FROM predictions WHERE id = %s AND user_id = %s", (record_id, user_id))
        db.commit()
        print(f"Deleted record with id: {record_id}")
        return redirect(url_for('records'))
    except mysql.connector.Error as e:
        db.rollback()
        print(f"Error deleting record: {e}")
        abort(500, description="Failed to delete record.")

if __name__ == '__main__':
    app.run(debug=True)