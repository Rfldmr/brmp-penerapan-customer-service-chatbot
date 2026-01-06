import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash

def init_database():
    """Initialize SQLite database and create tables"""
    db_path = 'chatbot_admin.db'
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create admins table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create sessions table for better session management
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            admin_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (admin_id) REFERENCES admins (id)
        )
    ''')
    
    # Create activity logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER,
            action TEXT NOT NULL,
            description TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES admins (id)
        )
    ''')
    
    # Create content settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_key TEXT UNIQUE NOT NULL,
            setting_value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by INTEGER,
            FOREIGN KEY (updated_by) REFERENCES admins (id)
        )
    ''')
    
    # Create visitor visits table for analytics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitor_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            page_path TEXT DEFAULT '/'
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_visits_date 
        ON visitor_visits(visited_at)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_visits_session 
        ON visitor_visits(session_id)
    ''')
    
    # Insert default content settings if not exist
    default_settings = [
        ('logo_dark', 'static/image/logo_dark.png'),
        ('logo_light', 'static/image/profil.png'),
        ('chatbot_name', 'BRMP Penerapan Chatbot'),
        ('subtitle', 'Selamat Datang di Chatbot BRMP Penerapan'),
        ('copyright_text', '© 2025 BRMP Penerapan. All Rights Reserved')
    ]
    
    for key, value in default_settings:
        cursor.execute('SELECT COUNT(*) FROM content_settings WHERE setting_key = ?', (key,))
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO content_settings (setting_key, setting_value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, datetime.now().isoformat()))
    
    # Check if default admin exists
    cursor.execute('SELECT COUNT(*) FROM admins WHERE username = ?', ('admin',))
    if cursor.fetchone()[0] == 0:
        # Create default admin if doesn't exist
        default_password_hash = generate_password_hash('admin123')
        cursor.execute('''
            INSERT INTO admins (username, password_hash, created_at)
            VALUES (?, ?, ?)
        ''', ('admin', default_password_hash, datetime.now().isoformat()))
        print("Default admin created: username='admin', password='admin123'")
    
    conn.commit()
    conn.close()
    print(f"Database initialized at: {os.path.abspath(db_path)}")

if __name__ == "__main__":
    init_database()