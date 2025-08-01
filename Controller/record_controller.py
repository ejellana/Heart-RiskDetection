# Controller/record_controller.py
from flask import Blueprint, render_template, request, redirect, url_for, session, abort
from config.database import DatabaseConfig
import json

record_bp = Blueprint('record', __name__, url_prefix='/record')

@record_bp.route('/')
def records():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    user_id = session['id']
    session.pop('records', None)
    conn = DatabaseConfig.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, prediction_data, model_type FROM predictions WHERE user_id = %s ORDER BY timestamp DESC", (user_id,))
    records = [{'id': row[0], 'data': json.loads(row[1]), 'model_type': row[2]} for row in cursor.fetchall()]
    conn.close()
    print(f"Fetched records count: {len(records)}")
    return render_template('records/records.html', records=records)

@record_bp.route('/delete_record/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    user_id = session['id']
    try:
        conn = DatabaseConfig.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = %s AND user_id = %s", (record_id, user_id))
        conn.commit()
        conn.close()
        print(f"Deleted record with id: {record_id}")
        return redirect(url_for('record.records'))
    except Exception as e:
        conn.rollback()
        conn.close()
        print(f"Error deleting record: {e}")
        abort(500, description="Failed to delete record.")