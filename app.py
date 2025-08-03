# app.py
from flask import Flask
from flask_cors import CORS
import config.database as db_config
from Controller.auth_controller import auth_bp
from Controller.prediction_controller import prediction_bp
from Controller.record_controller import record_bp
from Controller.android_api_controller import android_api_bp
from Model.implementations.heart_cluster_model import HeartClusterModel

app = Flask(__name__, template_folder='View', static_folder='static')
app.secret_key = db_config.DatabaseConfig.SECRET_KEY

# Enable CORS for Android app requests
CORS(app, resources={r"/api/android/*": {"origins": "*"}})

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(record_bp)
app.register_blueprint(android_api_bp, url_prefix='/api/android')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    