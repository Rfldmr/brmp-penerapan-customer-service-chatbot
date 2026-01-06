"""Database manager module for chatbot admin system.

This module provides DatabaseManager class for handling all database operations
including admin management, content settings, activity logging, and session management.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

from werkzeug.security import check_password_hash, generate_password_hash


class DatabaseManager:
    """Manage database operations for chatbot admin system."""
    
    def __init__(self, db_path='chatbot_admin.db'):
        """Initialize database manager.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database if it doesn't exist."""
        if not os.path.exists(self.db_path):
            from init_db import init_database
            init_database()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections.
        
        Yields:
            sqlite3.Connection: Database connection with row factory enabled
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # ===== Admin Management Methods =====
    
    def get_all_admins(self):
        """Get all active admin users.
        
        Returns:
            list: List of admin dictionaries
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, created_at, last_login, is_active
                FROM admins
                WHERE is_active = 1
                ORDER BY created_at DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]
    
    def get_admin_by_username(self, username):
        """Get admin by username.
        
        Args:
            username (str): Admin username
            
        Returns:
            dict or None: Admin data if found, None otherwise
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, password_hash, created_at, last_login, is_active
                FROM admins
                WHERE username = ? AND is_active = 1
            ''', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def create_admin(self, username, password):
        """Create new admin user.
        
        Args:
            username (str): Admin username
            password (str): Admin password (will be hashed)
            
        Returns:
            dict: Status and message
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                password_hash = generate_password_hash(password)
                cursor.execute('''
                    INSERT INTO admins (username, password_hash, created_at)
                    VALUES (?, ?, ?)
                ''', (username, password_hash, datetime.now().isoformat()))
                conn.commit()
                return {"status": "success", "message": f"Admin '{username}' berhasil ditambahkan"}
        except sqlite3.IntegrityError:
            return {"status": "error", "message": "Username sudah ada"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}
    
    def delete_admin(self, username):
        """Delete admin user (soft delete).
        
        Args:
            username (str): Admin username to delete
            
        Returns:
            dict: Status and message
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                admin = self.get_admin_by_username(username)
                if not admin:
                    return {"status": "error", "message": "Admin tidak ditemukan"}
                
                cursor.execute('SELECT COUNT(*) FROM admins WHERE is_active = 1')
                if cursor.fetchone()[0] <= 1:
                    return {"status": "error", "message": "Tidak dapat menghapus admin terakhir"}
                
                cursor.execute('''
                    UPDATE admins SET is_active = 0 WHERE username = ?
                ''', (username,))
                conn.commit()
                return {"status": "success", "message": f"Admin '{username}' berhasil dihapus"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}
    
    def verify_admin_credentials(self, username, password):
        """Verify admin login credentials.
        
        Args:
            username (str): Admin username
            password (str): Admin password
            
        Returns:
            dict or None: Admin data if credentials valid, None otherwise
        """
        admin = self.get_admin_by_username(username)
        if admin and check_password_hash(admin['password_hash'], password):
            self.update_last_login(admin['id'])
            return admin
        return None
    
    def update_last_login(self, admin_id):
        """Update last login timestamp.
        
        Args:
            admin_id (int): Admin ID
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE admins SET last_login = ? WHERE id = ?
            ''', (datetime.now().isoformat(), admin_id))
            conn.commit()
    
    def get_admin_count(self):
        """Get total number of active admins.
        
        Returns:
            int: Count of active admins
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM admins WHERE is_active = 1')
            return cursor.fetchone()[0]
    
    # ===== Content Management Methods =====
    
    def get_content_settings(self):
        """Get all content settings as dictionary.
        
        Returns:
            dict: Dictionary of setting_key: setting_value pairs
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT setting_key, setting_value FROM content_settings')
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    def get_content_setting(self, key):
        """Get specific content setting by key.
        
        Args:
            key (str): Setting key
            
        Returns:
            str or None: Setting value if found, None otherwise
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT setting_value FROM content_settings WHERE setting_key = ?', (key,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def update_content_setting(self, key, value, admin_id=None):
        """Update specific content setting.
        
        Args:
            key (str): Setting key
            value (str): New setting value
            admin_id (int, optional): Admin ID who made the change
            
        Returns:
            dict: Status and message
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE content_settings
                    SET setting_value = ?, updated_at = ?, updated_by = ?
                    WHERE setting_key = ?
                ''', (value, datetime.now().isoformat(), admin_id, key))
                conn.commit()
                return {"status": "success", "message": f"Setting '{key}' berhasil diupdate"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}
    
    def update_multiple_settings(self, settings_dict, admin_id=None):
        """Update multiple content settings at once.
        
        Args:
            settings_dict (dict): Dictionary of setting_key: value pairs
            admin_id (int, optional): Admin ID who made the change
            
        Returns:
            dict: Status and message
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().isoformat()
                for key, value in settings_dict.items():
                    cursor.execute('''
                        UPDATE content_settings
                        SET setting_value = ?, updated_at = ?, updated_by = ?
                        WHERE setting_key = ?
                    ''', (value, timestamp, admin_id, key))
                conn.commit()
                return {"status": "success", "message": "Settings berhasil diupdate"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}
    
    # ===== Activity Logging Methods =====
    
    def log_activity(self, admin_id, action, description=None, ip_address=None):
        """Log admin activity.
        
        Args:
            admin_id (int): Admin ID
            action (str): Action type
            description (str, optional): Action description
            ip_address (str, optional): IP address
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO activity_logs (admin_id, action, description, ip_address, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (admin_id, action, description, ip_address, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            print(f"Error logging activity: {e}")
    
    def get_recent_activities(self, limit=10):
        """Get recent admin activities.
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of activity dictionaries
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    al.action, al.description, al.created_at, al.ip_address, a.username
                FROM activity_logs al
                LEFT JOIN admins a ON al.admin_id = a.id
                ORDER BY al.created_at DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ===== Session Management Methods =====
    
    def create_session(self, session_id, admin_id, ip_address=None, user_agent=None):
        """Create admin session record.
        
        Args:
            session_id (str): Session ID
            admin_id (int): Admin ID
            ip_address (str, optional): IP address
            user_agent (str, optional): User agent string
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                expires_at = (datetime.now() + timedelta(hours=24)).isoformat()
                cursor.execute('''
                    INSERT INTO admin_sessions (session_id, admin_id, expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, admin_id, expires_at, ip_address, user_agent))
                conn.commit()
        except Exception as e:
            print(f"Error creating session: {e}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from database."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM admin_sessions WHERE expires_at < ?
                ''', (datetime.now().isoformat(),))
                conn.commit()
        except Exception as e:
            print(f"Error cleaning up sessions: {e}")
    
    # ===== Analytics Methods =====
    
    def record_visit(self, session_id, ip_address=None, user_agent=None, page_path='/'):
        """Record a visitor visit for analytics.
        
        Args:
            session_id (str): Visitor session ID
            ip_address (str, optional): IP address
            user_agent (str, optional): User agent string
            page_path (str): Page path visited
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO visitor_visits (session_id, ip_address, user_agent, visited_at, page_path)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, ip_address, user_agent, datetime.now().isoformat(), page_path))
                conn.commit()
        except Exception as e:
            print(f"Error recording visit: {e}")
    
    def get_visit_stats(self, period='day'):
        """Get visit statistics for specified period.
        
        Args:
            period (str): Time period - 'day', 'week', or 'month'
            
        Returns:
            dict: Statistics including total visits, unique visitors, and trend data
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate date range
            now = datetime.now()
            if period == 'day':
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                date_format = '%H:00'
                group_format = "strftime('%H', visited_at)"
            elif period == 'week':
                start_date = now - timedelta(days=7)
                date_format = '%Y-%m-%d'
                group_format = "date(visited_at)"
            else:  # month
                start_date = now - timedelta(days=30)
                date_format = '%Y-%m-%d'
                group_format = "date(visited_at)"
            
            # Get total visits
            cursor.execute('''
                SELECT COUNT(*) FROM visitor_visits 
                WHERE visited_at >= ?
            ''', (start_date.isoformat(),))
            total_visits = cursor.fetchone()[0]
            
            # Get unique visitors
            cursor.execute('''
                SELECT COUNT(DISTINCT session_id) FROM visitor_visits 
                WHERE visited_at >= ?
            ''', (start_date.isoformat(),))
            unique_visitors = cursor.fetchone()[0]
            
            # Get visits by time period
            cursor.execute(f'''
                SELECT 
                    {group_format} as period,
                    COUNT(*) as visits,
                    COUNT(DISTINCT session_id) as unique_visitors
                FROM visitor_visits 
                WHERE visited_at >= ?
                GROUP BY period
                ORDER BY period
            ''', (start_date.isoformat(),))
            
            trend_data = []
            for row in cursor.fetchall():
                period_label = row[0]
                trend_data.append({
                    'period': period_label,
                    'visits': row[1],
                    'unique_visitors': row[2]
                })
            
            return {
                'total_visits': total_visits,
                'unique_visitors': unique_visitors,
                'trend_data': trend_data,
                'period': period
            }
    
    def get_all_time_stats(self):
        """Get all-time visit statistics.
        
        Returns:
            dict: All-time statistics
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM visitor_visits')
            total_visits = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT session_id) FROM visitor_visits')
            unique_visitors = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT MIN(visited_at), MAX(visited_at) FROM visitor_visits
            ''')
            row = cursor.fetchone()
            first_visit = row[0] if row[0] else None
            last_visit = row[1] if row[1] else None
            
            return {
                'total_visits': total_visits,
                'unique_visitors': unique_visitors,
                'first_visit': first_visit,
                'last_visit': last_visit
            }


# Global database instance
db_manager = DatabaseManager()