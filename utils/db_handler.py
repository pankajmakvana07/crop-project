import streamlit as st
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
import bcrypt

load_dotenv()

# Database connection parameters
db_params = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# -------------------- Users --------------------
def verify_duplicate_user(email):
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count > 0

def authenticate_user(user_input, password):
    """
    Authenticate a user using email or phone number.
    Returns: (authenticated: bool, user_id: int or None)
    """
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    if user_input.isdigit():
        cur.execute("SELECT id, hash_password FROM users WHERE phonenumber = %s", (int(user_input),))
    else:
        cur.execute("SELECT id, hash_password FROM users WHERE email = %s", (user_input,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if not result:
        return False, None
    user_id, stored_hashed_password = result
    if bcrypt.checkpw(password.encode(), stored_hashed_password.encode()):
        return True, user_id
    else:
        return False, None

def save_user(email, password, extra_input_params):
    """
    Save a new user and return its id.
    """
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode('utf-8')
    
    columns = ['email', 'hash_password']
    values = [email, hashed_password]
    for key in extra_input_params.keys():
        columns.append(key)
        values.append(st.session_state.get(key, ''))
    
    columns_str = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(values))
    query = sql.SQL("INSERT INTO users ({}) VALUES ({}) RETURNING id").format(
        sql.SQL(columns_str),
        sql.SQL(placeholders)
    )
    
    cur.execute(query, values)
    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return user_id

def get_users():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()
    conn.close()
    return users

# -------------------- Soil Details --------------------
def save_user_detail(user_id, state, district, taluka, soil_type, ph):
    """
    Save soil details for a specific user into 'soil_details'.
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        query = """
            INSERT INTO soil_details (user_id, state, district, taluka, soil_type, pH)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (user_id, state, district, taluka, soil_type, float(ph)))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

def get_user_soil_details(user_id):
    """
    Retrieve the most recent soil details for a user.
    Returns: dict with soil parameters or None
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        query = """
            SELECT id, state, district, taluka, soil_type, pH
            FROM soil_details
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT 1
        """
        cur.execute(query, (user_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'state': result[1],
                'district': result[2],
                'taluka': result[3],
                'soil_type': result[4],
                'pH': float(result[5])
            }
        return None
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None

def get_all_user_soil_details(user_id):
    """
    Retrieve all soil detail records for a user.
    Returns: list of dicts
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        query = """
            SELECT id, state, district, taluka, soil_type, pH
            FROM soil_details
            WHERE user_id = %s
            ORDER BY id DESC
        """
        cur.execute(query, (user_id,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        return [{
            'id': row[0],
            'state': row[1],
            'district': row[2],
            'taluka': row[3],
            'soil_type': row[4],
            'pH': float(row[5])
        } for row in results]
    except Exception as e:
        st.error(f"Database Error: {e}")
        return []

def update_user_detail(detail_id, user_id, state, district, taluka, soil_type, ph):
    """
    Update existing soil details for a specific user.
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        query = """
            UPDATE soil_details 
            SET state = %s, district = %s, taluka = %s, soil_type = %s, pH = %s
            WHERE id = %s AND user_id = %s
        """
        cur.execute(query, (state, district, taluka, soil_type, float(ph), detail_id, user_id))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

def has_user_soil_details(user_id):
    """
    Check if user has any soil details.
    Returns: bool
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        query = "SELECT COUNT(*) FROM soil_details WHERE user_id = %s"
        cur.execute(query, (user_id,))
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count > 0
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

# -------------------- Prediction History --------------------
def save_prediction_history(user_id, prediction_type, district, taluka, soil_type, soil_ph, 
                            predicted_crop, prediction_result, confidence=None, predicted_yield=None):
    """
    Save prediction result to history.
    
    Args:
        user_id: User ID
        prediction_type: 'normal' or 'advanced'
        district: District name
        taluka: Taluka name
        soil_type: Soil type
        soil_ph: Soil pH value
        predicted_crop: Crop name (for advanced) or top crop (for normal)
        prediction_result: JSON string of full prediction result
        confidence: Confidence percentage (optional)
        predicted_yield: Predicted yield value (optional)
    
    Returns:
        prediction_id if successful, None otherwise
    """
    try:
        import json
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Ensure prediction_result is a JSON string
        if isinstance(prediction_result, dict):
            prediction_result = json.dumps(prediction_result)
        
        query = """
            INSERT INTO prediction_history 
            (user_id, prediction_type, district, taluka, soil_type, soil_ph, 
             predicted_crop, prediction_result, confidence, predicted_yield)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        cur.execute(query, (
            user_id, prediction_type, district, taluka, soil_type, float(soil_ph),
            predicted_crop, prediction_result, 
            float(confidence) if confidence is not None else None,
            float(predicted_yield) if predicted_yield is not None else None
        ))
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return prediction_id
    except Exception as e:
        print(f"Error saving prediction history: {e}")
        return None

def get_user_prediction_history(user_id, limit=None):
    """
    Retrieve prediction history for a user.
    
    Args:
        user_id: User ID
        limit: Maximum number of records to return (None for all)
    
    Returns:
        List of prediction history records
    """
    try:
        import json
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        query = """
            SELECT id, prediction_type, district, taluka, soil_type, soil_ph,
                   predicted_crop, prediction_result, confidence, predicted_yield, created_at
            FROM prediction_history
            WHERE user_id = %s
            ORDER BY created_at DESC
        """
        
        if limit:
            query += f" LIMIT {int(limit)}"
        
        cur.execute(query, (user_id,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        history = []
        for row in results:
            # Parse JSON result
            try:
                result_data = json.loads(row[7]) if isinstance(row[7], str) else row[7]
            except:
                result_data = {}
            
            history.append({
                'id': row[0],
                'prediction_type': row[1],
                'district': row[2],
                'taluka': row[3],
                'soil_type': row[4],
                'soil_ph': float(row[5]),
                'predicted_crop': row[6],
                'prediction_result': result_data,
                'confidence': float(row[8]) if row[8] is not None else None,
                'predicted_yield': float(row[9]) if row[9] is not None else None,
                'created_at': row[10]
            })
        
        return history
    except Exception as e:
        print(f"Error retrieving prediction history: {e}")
        return []

def get_prediction_stats(user_id):
    """
    Get statistics about user's predictions.
    
    Returns:
        dict with statistics
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Total predictions
        cur.execute("SELECT COUNT(*) FROM prediction_history WHERE user_id = %s", (user_id,))
        total = cur.fetchone()[0]
        
        # Most predicted crop
        cur.execute("""
            SELECT predicted_crop, COUNT(*) as count
            FROM prediction_history
            WHERE user_id = %s AND predicted_crop IS NOT NULL
            GROUP BY predicted_crop
            ORDER BY count DESC
            LIMIT 1
        """, (user_id,))
        most_predicted = cur.fetchone()
        
        # Recent predictions count (last 7 days)
        cur.execute("""
            SELECT COUNT(*) FROM prediction_history
            WHERE user_id = %s AND created_at >= NOW() - INTERVAL '7 days'
        """, (user_id,))
        recent = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        return {
            'total_predictions': total,
            'most_predicted_crop': most_predicted[0] if most_predicted else None,
            'most_predicted_count': most_predicted[1] if most_predicted else 0,
            'recent_predictions': recent
        }
    except Exception as e:
        print(f"Error getting prediction stats: {e}")
        return {
            'total_predictions': 0,
            'most_predicted_crop': None,
            'most_predicted_count': 0,
            'recent_predictions': 0
        }
