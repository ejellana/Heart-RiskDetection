# Controller/auth_controller.py
from flask import Blueprint, render_template, request, redirect, url_for, session
from config.database import DatabaseConfig
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/')
def home():
    return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    print(f"Request method: {request.method}")
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, name, password, specialty FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        print(f"User query result: {user}")
        conn.close()
        if user and check_password_hash(user[3], password):
            session['username'] = user[1]
            session['id'] = user[0]
            session['name'] = user[2]
            session['specialty'] = user[4]
            print(f"Session name set to: {session['name']}, specialty: {session['specialty']}")
            return redirect(url_for('prediction.model_selection'))
        else:
            return render_template('auth/login.html', error="Invalid username or password.")
    return render_template('auth/login.html')

@auth_bp.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('id', None)
    session.pop('name', None)
    session.pop('specialty', None)
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        specialty = request.form['specialty']
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            conn.close()
            return render_template('auth/register.html', error="Username already taken.")
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        sql = "INSERT INTO users (name, username, password, specialty) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (name, username, hashed_password, specialty))
        conn.commit()
        conn.close()
        return redirect(url_for('auth.login') + '?success=true')
    return render_template('auth/register.html')