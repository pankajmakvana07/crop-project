"""
Migration script to add prediction_history table to existing database
Run this if you already have a database and need to add the new table
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Database connection parameters
db_params = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}


def add_prediction_history_table():
    """Add prediction_history table to existing database"""
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Check if table already exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'prediction_history'
            );
        """)
        
        table_exists = cur.fetchone()[0]
        
        if table_exists:
            print("✓ prediction_history table already exists")
            cur.close()
            conn.close()
            return
        
        # Create the table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id SERIAL PRIMARY KEY,
                user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                prediction_type VARCHAR(50) NOT NULL,
                district VARCHAR(255) NOT NULL,
                taluka VARCHAR(255) NOT NULL,
                soil_type VARCHAR(255) NOT NULL,
                soil_ph NUMERIC(4,2) NOT NULL,
                predicted_crop VARCHAR(255),
                prediction_result TEXT NOT NULL,
                confidence NUMERIC(5,2),
                predicted_yield NUMERIC(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for faster queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_history_user_id 
            ON prediction_history(user_id);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at 
            ON prediction_history(created_at DESC);
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        print("✓ prediction_history table created successfully")
        print("✓ Indexes created for better performance")
        
    except psycopg2.Error as e:
        print(f"✗ Error creating prediction_history table: {e}")
        raise


if __name__ == "__main__":
    print("Adding prediction_history table to database...")
    add_prediction_history_table()
    print("\nMigration completed successfully!")
    print("You can now use the prediction history feature.")
