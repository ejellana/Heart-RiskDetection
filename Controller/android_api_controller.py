from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from config.database import DatabaseConfig
from flask_cors import cross_origin
import logging
import json
import numpy as np
from Model.implementations.heart_cluster_model import HeartClusterModel
from Model.implementations.neural_network_model import NeuralNetworkModel

logging.basicConfig(level=logging.DEBUG)
android_api_bp = Blueprint('android_api', __name__)

# Allowed specialties for registration
ALLOWED_SPECIALTIES = [
    "Cardiologist",
    "Interventional Cardiologist",
    "Electrophysiologist",
    "Cardiac Surgeon",
    "General Practitioner (GP)"
]

@android_api_bp.route('/register', methods=['POST'])
@cross_origin()
def register():
    data = request.form
    name = data.get('name')
    username = data.get('username')
    password = data.get('password')
    specialty = data.get('specialty', '')

    logging.debug(f"Register request: name={name}, username={username}, specialty={specialty}")

    # Validate inputs
    if not all([name, username, password]):
        response = {"message": "Missing name, username, or password"}
        logging.debug(f"Register response: {json.dumps(response)}")
        return jsonify(response), 400
    
    if specialty and specialty not in ALLOWED_SPECIALTIES:
        response = {"message": f"Invalid specialty. Must be one of: {', '.join(ALLOWED_SPECIALTIES)}"}
        logging.debug(f"Register response: {json.dumps(response)}")
        return jsonify(response), 400

    try:
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            conn.close()
            response = {"message": "Username already taken"}
            logging.debug(f"Register response: {json.dumps(response)}")
            return jsonify(response), 400

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        sql = "INSERT INTO users (name, username, password, specialty) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (name, username, hashed_password, specialty or None))
        conn.commit()
        conn.close()
        
        response = {"username": username, "name": name, "specialty": specialty}
        logging.debug(f"Register response: {json.dumps(response)}")
        return jsonify(response), 201
    
    except Exception as e:
        response = {"message": f"Registration failed: {str(e)}"}
        logging.error(f"Registration error: {str(e)}, Response: {json.dumps(response)}")
        return jsonify(response), 500

@android_api_bp.route('/login', methods=['POST'])
@cross_origin()
def login():
    data = request.form
    username = data.get('username')
    password = data.get('password')

    logging.debug(f"Login request: username={username}")

    if not all([username, password]):
        response = {"message": "Missing username or password"}
        logging.debug(f"Login response: {json.dumps(response)}")
        return jsonify(response), 400

    try:
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password, name FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            response = {"id": str(user[0]), "username": user[1], "name": user[3], "message": "Login successful"}
            logging.debug(f"Login response: {json.dumps(response)}")
            return jsonify(response), 200
        else:
            response = {"message": "Invalid username or password"}
            logging.debug(f"Login response: {json.dumps(response)}")
            return jsonify(response), 401

    except Exception as e:
        response = {"message": f"Login failed: {str(e)}"}
        logging.error(f"Login error: {str(e)}, Response: {json.dumps(response)}")
        return jsonify(response), 500

@android_api_bp.route('/records/<user_id>', methods=['GET'])
@cross_origin()
def get_records(user_id):
    try:
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.id, p.user_id, p.prediction_data, p.timestamp, p.model_type, u.username
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            WHERE p.user_id = %s
        """, (user_id,))
        records = cursor.fetchall()
        conn.close()

        predictions = [
            {
                "id": record[0],
                "user_id": str(record[1]),  # Convert INT to string
                "username": record[5],
                "prediction_data": record[2],
                "timestamp": record[3].isoformat(),
                "model_type": record[4]
            } for record in records
        ]
        
        logging.debug(f"Get records response: {json.dumps(predictions)}")
        return jsonify(predictions), 200

    except Exception as e:
        response = {"message": f"Failed to fetch records: {str(e)}"}
        logging.error(f"Records fetch error: {str(e)}, Response: {json.dumps(response)}")
        return jsonify(response), 500

@android_api_bp.route('/records/all', methods=['GET'])
@cross_origin()
def get_all_records():
    try:
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.id, p.user_id, p.prediction_data, p.timestamp, p.model_type, u.username
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.timestamp
        """)
        records = cursor.fetchall()
        conn.close()

        predictions = [
            {
                "id": record[0],
                "user_id": str(record[1]),  # Convert INT to string
                "username": record[5],
                "prediction_data": record[2],
                "timestamp": record[3].isoformat(),
                "model_type": record[4]
            } for record in records
        ]
        
        logging.debug(f"Get all records response: {json.dumps(predictions)}")
        return jsonify(predictions), 200

    except Exception as e:
        response = {"message": f"Failed to fetch all records: {str(e)}"}
        logging.error(f"Get all records error: {str(e)}, Response: {json.dumps(response)}")
        return jsonify(response), 500

@android_api_bp.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.form.to_dict()
    user_id = data.get('user_id')
    model_type = data.get('model_type')
    input_data = {
        "age": data.get('age', None),
        "trestbps": data.get('trestbps', None),
        "chol": data.get('chol', None),
        "thalach": data.get('thalach', None),
        "oldpeak": data.get('oldpeak', None),
        "ca": data.get('ca', None),
        "sex": data.get('sex', None),
        "cp": data.get('cp', None),
        "exang": data.get('exang', None),
        "slope": data.get('slope', None),
        "thal": data.get('thal', None)
    }

    logging.debug(f"Predict request: user_id={user_id}, model_type={model_type}, input_data={input_data}")

    # Define required fields per model type
    required_fields = {
        "Cluster": ["age", "chol", "thalach", "oldpeak", "ca", "sex", "cp", "thal"],  # Minimal required fields
        "Neural Network": ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "sex", "cp", "exang", "slope", "thal"]
    }
    # Check for missing required fields
    missing_fields = [key for key in required_fields.get(model_type, []) if input_data[key] is None or (isinstance(input_data[key], str) and not input_data[key].strip())]
    invalid_types = []
    for key in required_fields.get(model_type, []):
        try:
            if input_data[key] is not None and input_data[key].strip():
                if key in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
                    input_data[key] = float(input_data[key])
                elif key in ['ca', 'sex', 'cp', 'exang', 'slope', 'thal']:
                    input_data[key] = int(input_data[key])
        except (ValueError, TypeError):
            invalid_types.append(key)

    if not user_id or not model_type or missing_fields or invalid_types:
        error_msg = {
            "message": "Invalid or missing fields",
            "missing_fields": missing_fields,
            "invalid_types": invalid_types,
            "user_id": user_id,
            "model_type": model_type
        }
        logging.error(f"Validation error: {error_msg}")
        return jsonify(error_msg), 400

    try:
        # Pass input_data dictionary to model
        logging.debug(f"Input data for model: {input_data}")

        # Load model and make prediction
        if model_type == "Cluster":
            model = HeartClusterModel()
            prediction = model.Predict(input_data)
            cluster = prediction['cluster']
            risk_level = prediction['risk_level']
        elif model_type == "Neural Network":
            model = NeuralNetworkModel()
            prediction = model.Predict(input_data)
            cluster = prediction['cluster']
            risk_level = prediction['risk_level']
        else:
            response = {"message": "Invalid model_type: must be 'Cluster' or 'Neural Network'"}
            logging.debug(f"Predict response: {json.dumps(response)}")
            return jsonify(response), 400

        # Create prediction data
        prediction_data = {
            "cluster": cluster,
            "risk_level": risk_level,
            "message": f"Patient is predicted to be in Cluster {cluster} ({risk_level}).",
            "input": input_data,
            "model_type": model_type
        }

        # Save to database
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        sql = "INSERT INTO predictions (user_id, prediction_data, model_type) VALUES (%s, %s, %s)"
        cursor.execute(sql, (user_id, json.dumps(prediction_data), model_type))
        conn.commit()
        conn.close()

        logging.debug(f"Predict response: {json.dumps(prediction_data)}")
        return jsonify(prediction_data), 200

    except Exception as e:
        response = {"message": f"Prediction failed: {str(e)}"}
        logging.error(f"Prediction error: {str(e)}, Response: {json.dumps(response)}")
        return jsonify(response), 500