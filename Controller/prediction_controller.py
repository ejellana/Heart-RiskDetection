# Controller/prediction_controller.py
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
from Model.implementations.heart_cluster_model import HeartClusterModel
from Model.implementations.neural_network_model import NeuralNetworkModel
import tensorflow as tf
import numpy as np
import json
from config.database import DatabaseConfig

prediction_bp = Blueprint('prediction', __name__, url_prefix='/prediction')

cluster_model = HeartClusterModel()
neural_model = NeuralNetworkModel()

@prediction_bp.route('/', methods=['GET', 'POST'])
def model_selection():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    if request.method == 'POST':
        model_type = request.form['model_type']
        if model_type == 'cluster':
            return redirect(url_for('prediction.predict'))
        elif model_type == 'neural':
            return redirect(url_for('prediction.neural_input'))
    return render_template('prediction/model_selection.html', name=session.get('name', 'Guest'), specialty=session.get('specialty', 'Specialist'))

@prediction_bp.route('/predict')
def predict():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    return render_template('prediction/index.html')

@prediction_bp.route('/neural_input')
def neural_input():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    return render_template('prediction/neural_input.html', name=session.get('name', 'Guest'), specialty=session.get('specialty', 'Specialist'))

@prediction_bp.route('/submit', methods=['POST'])
def submit():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
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
                response = result
            except ValueError as ve:
                return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
        else:  # neural network
            required_fields = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'sex', 'cp', 'exang', 'slope', 'thal']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
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
                response = result
            except ValueError as ve:
                return jsonify({'error': f'Invalid input: {str(ve)}'}), 400

        session['result'] = response
        print(f"Response set in session: {response}")
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        user_id = session['id']
        prediction_data = json.dumps(response)
        cursor.execute(
            "INSERT INTO predictions (user_id, prediction_data, model_type, timestamp) VALUES (%s, %s, %s, NOW())",
            (user_id, prediction_data, response['model_type'])
        )
        conn.commit()
        conn.close()

        print(f"Response JSON: {json.dumps(response)}")
        return redirect(url_for('prediction.result'))
    except ValueError as ve:
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@prediction_bp.route('/result')
def result():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    result = session.get('result')
    if not result:
        print("Warning: No result data in session")
        return redirect(url_for('prediction.model_selection'))
    session.pop('result', None)
    print(f"Result from session: {result}")
    return render_template('prediction/results.html', result=result, model_type=result.get('model_type', 'Cluster'))