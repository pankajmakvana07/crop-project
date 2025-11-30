"""
Quick check to see if prediction_history table exists
Run this to determine if you need to run the migration
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

db_params = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def check_table():
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'prediction_history'
            );
        """)
        
        exists = cur.fetchone()[0]
        
        print("=" * 60)
        print("PREDICTION HISTORY TABLE CHECK")
        print("=" * 60)
        
        if exists:
            # Get row count
            cur.execute("SELECT COUNT(*) FROM prediction_history;")
            count = cur.fetchone()[0]
            
            print("✅ Table EXISTS")
            print(f"   Current records: {count}")
            print("\n📌 ACTION NEEDED:")
            print("   ❌ DO NOT run add_prediction_history_table.py")
            print("   ✅ You're ready to use the feature!")
        else:
            print("❌ Table DOES NOT EXIST")
            print("\n📌 ACTION NEEDED:")
            print("   ✅ Run: python db/add_prediction_history_table.py")
            print("   (Run it ONCE to create the table)")
        
        print("=" * 60)
        
        cur.close()
        conn.close()
        
    except psycopg2.OperationalError as e:
        print("❌ Cannot connect to database")
        print(f"   Error: {e}")
        print("\n📌 ACTION NEEDED:")
        print("   1. Check if PostgreSQL is running")
        print("   2. Verify .env file has correct credentials")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_table()
