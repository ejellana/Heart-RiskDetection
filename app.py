# app.py
from flask import Flask
import config.database as db_config
from Controller.auth_controller import auth_bp
from Controller.prediction_controller import prediction_bp
from Controller.record_controller import record_bp
from Model.implementations.heart_cluster_model import HeartClusterModel

app = Flask(__name__, template_folder='View', static_folder='static')
app.secret_key = db_config.DatabaseConfig.SECRET_KEY

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(record_bp)

if __name__ == '__main__':
    app.run(debug=True)
    