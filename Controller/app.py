from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from Model.heart_cluster_model import HeartClusterModel
import joblib
import os
import json
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder='../view', static_folder='../static')
app.secret_key = 'your_secret_key'  # Change this to a secure random key in production

# Load pre-trained model with explicit paths (relative to project root)
model = HeartClusterModel()
model_dir = os.path.join(os.path.dirname(__file__), '..')  # Go up to project root
model.model = joblib.load(os.path.join(model_dir, 'heart_cluster_model.pkl'))
model.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
model.imputer_numeric = joblib.load(os.path.join(model_dir, 'imputer_numeric.pkl'))
model.imputer_categorical = joblib.load(os.path.join(model_dir, 'imputer_categorical.pkl'))

# MySQL Configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Leave empty for XAMPP default, or set your MySQL password
    database="heartPrediction"
)
cursor = db.cursor()

@app.route('/')
def home():
    return redirect(url_for('login'))  # Always redirect to login as the entry point

@app.route('/login', methods=['GET', 'POST'])
def login():
    print(f"Request method: {request.method}")  # Debug print
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT id, username, name, password, specialty FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        print(f"User query result: {user}")  # Debug print
        if user and check_password_hash(user[3], password):
            session['username'] = user[1]
            session['id'] = user[0]
            session['name'] = user[2]  # Store the doctor's name
            session['specialty'] = user[4]  # Store the doctor's specialty
            print(f"Session name set to: {session['name']}, specialty: {session['specialty']}")  # Debug print
            return redirect(url_for('predict'))  # Redirect to prediction page after login
        else:
            return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/predict')
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        specialty = request.form['specialty']

        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return render_template('register.html', error="Username already taken.")

        # Insert new user with hashed password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        sql = "INSERT INTO users (name, username, password, specialty) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (name, username, hashed_password, specialty))
        db.commit()
        return redirect(url_for('login') + '?success=true')

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('id', None)
    session.pop('name', None)  # Clear the name from the session
    session.pop('specialty', None)  # Clear the specialty from the session
    return render_template('login.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'username' not in session:
        return redirect(url_for('login'))
    try:
        data = request.form
        features = {
            'age': float(data['age']),
            'sex': int(data['sex']),
            'cp': int(data['cp']),
            'chol': float(data['chol']),
            'thalach': float(data['thalach']),
            'oldpeak': float(data['oldpeak']),
            'ca': int(data['ca']),
            'thal': int(data['thal']),
            'trestbps': model.imputer_numeric.statistics_[1],
            'exang': model.imputer_categorical.statistics_[2],
            'slope': model.imputer_categorical.statistics_[3]
        }
        result = model.predict(features)
        cluster = int(float(str(result['cluster']).replace('cluster', ''))) if isinstance(result['cluster'], str) and 'cluster' in str(result['cluster']).lower() else int(result['cluster'])
        print(f"Debug: Predicted result = {result}")
        response = {
            'cluster': cluster,
            'risk_level': result['risk_level'],
            'message': f'Patient is predicted to be in Cluster {cluster} ({result["risk_level"]}).',
            'input': {
                'age': float(data['age']),
                'chol': float(data['chol']),
                'thalach': float(data['thalach']),
                'oldpeak': float(data['oldpeak']),
                'ca': int(data['ca']),
                'thal': int(data['thal'])
            }
        }
        print(f"Response JSON: {json.dumps(response)}")  # Debug print
        session['result'] = response  # Store in session
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
    session.pop('result', None)  # Clear session after retrieval
    print(f"Received result from session: {result}")  # Debug print
    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)