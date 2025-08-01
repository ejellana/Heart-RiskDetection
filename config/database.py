# config/database.py
import mysql.connector
import os

class DatabaseConfig:
    HOST = "localhost"
    USER = "root"
    PASSWORD = ""  # Adjust for your MySQL setup
    DATABASE = "heartPrediction"

    @staticmethod
    def get_connection():
        return mysql.connector.connect(
            host=DatabaseConfig.HOST,
            user=DatabaseConfig.USER,
            password=DatabaseConfig.PASSWORD,
            database=DatabaseConfig.DATABASE
        )

    SECRET_KEY = os.urandom(24)  # For Flask session security