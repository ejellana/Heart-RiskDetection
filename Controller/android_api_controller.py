from flask import Blueprint, request, jsonify
from Model.implementations.heart_cluster_model import HeartClusterModel
from Model.implementations.neural_network_model import NeuralNetworkModel
from config.database import DatabaseConfig
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api/android')

cluster_model = HeartClusterModel()
neural_model = NeuralNetworkModel()

@api_bp.route('/')
def index():
    return jsonify({'message': 'Android API is running'})

@api_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.form
        name = data.get('name')
        username = data.get('username')
        password = data.get('password')
        specialty = data.get('specialty')

        logger.debug(f"Register request: name={name}, username={username}, specialty={specialty}")

        if not all([name, username, password]):
            return jsonify({'error': 'Missing required fields'}), 400

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'Username already exists'}), 400

        cursor.execute(
            "INSERT INTO users (name, username, password, specialty) VALUES (%s, %s, %s, %s)",
            (name, username, hashed_password, specialty)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        response = {
            'id': str(user_id),  # Convert to string to match /login response
            'username': username,
            'name': name,
            'specialty': specialty
        }
        logger.debug(f"Register response: {response}")
        return jsonify(response), 201
    except Exception as e:
        logger.error(f"Register error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.form
        username = data.get('username')
        password = data.get('password')

        if not all([username, password]):
            return jsonify({'error': 'Missing username or password'}), 400

        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, name, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            logger.debug(f"Login response: {{'id': '{user[0]}', 'username': '{user[1]}', 'name': '{user[2]}', 'message': 'Login successful'}}")
            return jsonify({
                'id': str(user[0]),
                'username': user[1],
                'name': user[2],
                'message': 'Login successful'
            }), 200
        else:
            logger.error(f"Login failed for username: {username}")
            return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        user_id = data.get('user_id')
        model_type = data.get('model_type')

        logger.debug(f"Predict request: user_id={user_id}, model_type={model_type}, input_data={dict(data)}")

        if not all([user_id, model_type]):
            return jsonify({'error': 'Missing user_id or model_type'}), 400

        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            logger.error(f"User not found: user_id={user_id}")
            return jsonify({'error': 'User not found'}), 404

        response = {}
        if model_type == 'Cluster':
            required_fields = ['age', 'sex', 'cp', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields for Cluster model'}), 400
            try:
                features = {
                    'age': float(data['age']),
                    'sex': int(data['sex']),
                    'cp': int(data['cp']),
                    'trestbps': float(data.get('trestbps', 0.0)),
                    'chol': float(data['chol']),
                    'thalach': float(data['thalach']),
                    'exang': int(data.get('exang', 0)),
                    'oldpeak': float(data['oldpeak']),
                    'slope': int(data.get('slope', 0)),
                    'ca': int(data['ca']),
                    'thal': int(data['thal'])
                }
                result = cluster_model.Predict(features)
                response = {
                    'cluster': int(result['cluster']),
                    'risk_level': result['risk_level'],
                    'message': result['message'],
                    'input': {key: str(value) for key, value in features.items()},
                    'model_type': 'Cluster'
                }
            except ValueError as ve:
                logger.error(f"Cluster prediction error: {str(ve)}")
                return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
        else:  # Neural Network
            required_fields = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'sex', 'cp', 'exang', 'slope', 'thal']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields for Neural Network model'}), 400
            try:
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
                result = neural_model.Predict(features)
                response = {
                    'cluster': int(result['cluster']),
                    'risk_level': result['risk_level'],
                    'message': result['message'],
                    'input': {key: str(value) for key, value in features.items()},
                    'model_type': 'Neural Network'
                }
            except ValueError as ve:
                logger.error(f"Neural Network prediction error: {str(ve)}")
                return jsonify({'error': f'Invalid input: {str(ve)}'}), 400

        prediction_data = json.dumps(response)
        cursor.execute(
            "INSERT INTO predictions (user_id, prediction_data, model_type, timestamp) VALUES (%s, %s, %s, NOW())",
            (user_id, prediction_data, response['model_type'])
        )
        conn.commit()
        conn.close()

        logger.debug(f"Predict response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Predict error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/records/<user_id>')
def get_records(user_id):
    try:
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT p.id, p.user_id, u.username, p.prediction_data, p.model_type, p.timestamp "
            "FROM predictions p JOIN users u ON p.user_id = u.id WHERE p.user_id = %s",
            (user_id,)
        )
        records = cursor.fetchall()
        conn.close()
        response = [{
            'id': record[0],
            'user_id': str(record[1]),
            'username': record[2],
            'prediction_data': record[3],
            'model_type': record[4],
            'timestamp': record[5].isoformat()
        } for record in records]
        logger.debug(f"Get records response for user_id={user_id}: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Get records error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/records/all')
def get_all_records():
    try:
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT p.id, p.user_id, u.username, p.prediction_data, p.model_type, p.timestamp "
            "FROM predictions p JOIN users u ON p.user_id = u.id"
        )
        records = cursor.fetchall()
        conn.close()
        response = [{
            'id': record[0],
            'user_id': str(record[1]),
            'username': record[2],
            'prediction_data': record[3],
            'model_type': record[4],
            'timestamp': record[5].isoformat()
        } for record in records]
        logger.debug(f"Get all records response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Get all records error: {str(e)}")
        return jsonify({'error': str(e)}), 400