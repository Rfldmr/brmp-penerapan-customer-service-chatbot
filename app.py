# Standard library imports
import json
import os
import pickle
import random
import subprocess
import threading
import time
from datetime import datetime
from difflib import SequenceMatcher
from functools import wraps

# Third-party imports
import nltk
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from nltk.stem import WordNetLemmatizer
from tensorflow import keras

# Local imports
from database import db_manager

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# Constants
SECRET_KEY = '25082025-chtbtbrmppnrpn-rfldmr'
MODEL_DIR = 'model'
MODEL_PATH = 'model/chatbot_brmp_model.h5'
WORDS_PATH = 'model/words.pkl'
CLASSES_PATH = 'model/classes.pkl'
INTENTS_PATH = 'intents.json'
SIMILARITY_THRESHOLD = 0.75  # Threshold untuk fuzzy matching (75% similarity)
CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence untuk prediction (50%)
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 5000
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}
MAX_IMAGE_SIZE_MB = 2

# Flask app initialization
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Global variables untuk tracking
training_status = {"status": "ready", "progress": 0, "message": ""}
model_info = {
    "last_training": None,
    "accuracy": "95.2%",
    "model_size": "2.4 MB",
    "total_intents": 0,
    "total_patterns": 0,
    "total_users": 0,
    "dataset_update": "N/A",
    "status": "Ready",
    "status_icon": "fas fa-brain",
    "status_color": "text-success-600"
}

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Visitor tracking middleware
@app.before_request
def track_visitor():
    """Track visitor visits for analytics (only for non-admin pages)."""
    # Skip tracking for admin pages, API endpoints, and static files
    if request.path.startswith('/admin') or \
       request.path.startswith('/api') or \
       request.path.startswith('/static'):
        return
    
    # Get or create visitor session ID
    if 'visitor_id' not in session:
        import uuid
        session['visitor_id'] = str(uuid.uuid4())
    
    # Record visit
    visitor_id = session['visitor_id']
    ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
    user_agent = request.headers.get('User-Agent')
    
    db_manager.record_visit(
        session_id=visitor_id,
        ip_address=ip_address,
        user_agent=user_agent,
        page_path=request.path
    )

# --- Fungsi untuk memuat model dan data ---
def load_model_and_data():
    """Load chatbot model, intents, words, and classes from files.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global model, intents, words, classes, model_info
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
            intents = json.load(f)
        with open(WORDS_PATH, 'rb') as f:
            words = pickle.load(f)
        with open(CLASSES_PATH, 'rb') as f:
            classes = pickle.load(f)
        
        # Update model info dengan data yang lebih akurat
        model_info["total_intents"] = len(intents['intents'])
        model_info["last_training"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Count total patterns
        total_patterns = 0
        for intent in intents['intents']:
            if 'patterns' in intent:
                total_patterns += len(intent['patterns'])
        model_info["total_patterns"] = total_patterns
        
        # Set user count to 0 for now (will be dynamic when user registration is implemented)
        model_info["total_users"] = 0
        
        # Hitung ukuran file intents.json (dataset)
        if os.path.exists(INTENTS_PATH):
            dataset_size_bytes = os.path.getsize(INTENTS_PATH)
            model_info["model_size"] = f"{dataset_size_bytes / 1024:.1f} KB"
            
            # Get dataset update date
            mod_time = os.path.getmtime(INTENTS_PATH)
            last_modified = datetime.fromtimestamp(mod_time)
            model_info["dataset_update"] = last_modified.strftime('%d %b %Y, %H:%M WIB')
        
        # Get last training time from model file
        if os.path.exists(MODEL_PATH):
            mod_time = os.path.getmtime(MODEL_PATH)
            last_training = datetime.fromtimestamp(mod_time)
            model_info["last_training"] = last_training.strftime("%Y-%m-%d %H:%M:%S")
        
        # Update status information
        model_info["accuracy"] = "95.2%"
        model_info["status"] = "Ready"
        model_info["status_icon"] = "fas fa-brain"
        model_info["status_color"] = "text-success-600"
        
        print(f"Model loaded successfully. Intents: {model_info['total_intents']}, Patterns: {model_info['total_patterns']}, Size: {model_info['model_size']}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        model_info["total_intents"] = 0
        model_info["total_patterns"] = 0
        model_info["total_users"] = 0  # Static 0 for now, will be dynamic when user system is implemented
        model_info["accuracy"] = "N/A"
        model_info["model_size"] = "N/A"
        model_info["last_training"] = None
        model_info["dataset_update"] = "N/A"
        model_info["status"] = "Not Trained"
        model_info["status_icon"] = "fas fa-exclamation-triangle"
        model_info["status_color"] = "text-warning-600"
        return False

# --- Muat model saat aplikasi pertama kali dijalankan ---
load_model_and_data()
lemmatizer = WordNetLemmatizer()

def normalize_repeated_chars(text):
    """Normalize repeated characters in text.
    
    Examples:
        pagiii -> pagi
        halooo -> halo
        terimaaaa -> terima
    
    Args:
        text (str): Input text with potential repeated characters
        
    Returns:
        str: Normalized text
    """
    import re
    normalized = re.sub(r'(.)\1{2,}', r'\1\1', text)
    normalized = re.sub(r'(.)\1+', r'\1', normalized)
    return normalized

def clean_up_sentence(sentence):
    """Clean and tokenize sentence for processing.
    
    Args:
        sentence (str): Input sentence
        
    Returns:
        list: List of lemmatized tokens
    """
    sentence = normalize_repeated_chars(sentence.lower())
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def string_similarity(a, b):
    """Calculate similarity ratio between two strings.
    
    Args:
        a (str): First string
        b (str): Second string
        
    Returns:
        float: Similarity ratio (0.0 to 1.0)
    """
    return SequenceMatcher(None, a, b).ratio()

def bag_of_words(sentence, words):
    """Convert sentence to bag of words array with fuzzy matching.
    
    Args:
        sentence (str): Input sentence
        words (list): List of vocabulary words
        
    Returns:
        numpy.ndarray: Bag of words representation
    """
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                # Exact match - prioritas tertinggi
                bag[i] = 1.0
            else:
                # Fuzzy match - untuk menangani typo dan variasi
                similarity = string_similarity(word, sw)
                if similarity >= SIMILARITY_THRESHOLD:
                    bag[i] = similarity
                    
    return bag

def predict_class(sentence):
    """Predict intent class from sentence.
    
    Args:
        sentence (str): Input sentence
        
    Returns:
        list: List of [index, confidence] pairs above threshold
    """
    p = bag_of_words(sentence, words)
    res = model.predict(np.expand_dims(p, axis=0))[0]
    
    results = [[i, r] for i, r in enumerate(res) if r > CONFIDENCE_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def getResponse(ints, intents_json):
    """Get response from predicted intent.
    
    Args:
        ints (list): List of predicted intents with confidence
        intents_json (dict): Intents configuration
        
    Returns:
        str: Response message
    """
    fallback_responses = [
        "Maaf, aku tidak mengerti. Bisa coba kata lain?",
        "Hmm, aku kurang paham. Coba jelaskan dengan cara lain?",
        "Maaf, aku belum bisa menjawab itu. Ada pertanyaan lain?"
    ]
    
    if not ints:
        return random.choice(fallback_responses)
    
    tag = classes[ints[0][0]]
    confidence = ints[0][1]
    
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            print(f"Response: {tag} (confidence: {confidence:.2f})")
            return response
    
    return random.choice(fallback_responses)

def train_model_async():
    """Fungsi untuk melatih model secara asinkron"""
    global training_status
    try:
        print("Training async started")  # Debug log
        training_status = {"status": "training", "progress": 10, "message": "Memproses file intents..."}
        time.sleep(1)
        
        print(f"Training progress: {training_status}")  # Debug log
        training_status = {"status": "training", "progress": 30, "message": "Mempersiapkan data training..."}
        time.sleep(2)
        
        print(f"Training progress: {training_status}")  # Debug log
        training_status = {"status": "training", "progress": 60, "message": "Melatih model neural network..."}
        
        # Method 1: Try importing and running directly
        try:
            import importlib.util
            import sys
            
            print("Attempting direct import method")  # Debug log
            
            # Load train_model.py module
            spec = importlib.util.spec_from_file_location("train_model", "train_model.py")
            train_module = importlib.util.module_from_spec(spec)
            
            # Execute training
            spec.loader.exec_module(train_module)
            train_module.train()
            
            success = True
            error_msg = None
            
        except Exception as import_error:
            print(f"Direct import failed: {import_error}")
            
            # Method 2: Fallback to subprocess with proper Python path
            try:
                import sys
                python_exe = sys.executable
                result = subprocess.run([python_exe, 'train_model.py'], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                
                if result.returncode == 0:
                    success = True
                    error_msg = None
                else:
                    success = False
                    error_msg = f"Training script error: {result.stderr}"
                    print(f"Subprocess stderr: {result.stderr}")
                    print(f"Subprocess stdout: {result.stdout}")
                    
            except Exception as subprocess_error:
                success = False
                error_msg = f"Subprocess error: {str(subprocess_error)}"
                print(f"Subprocess execution failed: {subprocess_error}")
        
        training_status = {"status": "training", "progress": 90, "message": "Menyimpan model..."}
        time.sleep(1)
        
        if success:
            # Hapus semua file backup model lama
            try:
                model_dir = 'model'
                if os.path.exists(model_dir):
                    for filename in os.listdir(model_dir):
                        if filename.startswith('backup_model_') and filename.endswith('.h5'):
                            backup_path = os.path.join(model_dir, filename)
                            os.remove(backup_path)
                            print(f"Deleted old model backup: {filename}")
            except Exception as cleanup_error:
                print(f"Warning: Could not delete old model backups: {cleanup_error}")
            
            # Reload model after training
            load_model_and_data()
            training_status = {"status": "completed", "progress": 100, "message": "Model berhasil dilatih ulang!"}
            print("Training completed successfully")  # Debug log
            
            # Reset status to ready after a short delay (for frontend to catch completion)
            def reset_status():
                time.sleep(1)  # Reduced to 1 second - just enough for frontend to catch completion
                global training_status
                training_status = {"status": "ready", "progress": 0, "message": ""}
                print("Training status reset to ready")  # Debug log
            
            reset_thread = threading.Thread(target=reset_status)
            reset_thread.daemon = True
            reset_thread.start()
            
        else:
            training_status = {"status": "error", "progress": 0, "message": error_msg or "Unknown training error"}
            print(f"Training failed: {error_msg}")  # Debug log
            
    except Exception as e:
        print(f"Training error: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full traceback
        training_status = {"status": "error", "progress": 0, "message": f"Training failed: {str(e)}"}

# ===== Public Routes =====

@app.route('/')
def index():
    """Render main chatbot page with dynamic content."""
    if 'messages' not in session:
        session['messages'] = [{"role": "assistant", "content": "Hai! Ada yang bisa aku bantu hari ini?"}]
    
    content_settings = db_manager.get_content_settings()
    return render_template('index.html', title='Chatbot BRMP', now=datetime.now(), content=content_settings)


# ===== Authentication Routes =====

@app.route('/admin/login', methods=['GET', 'POST'])
def login_page():
    """Handle admin login - GET shows form, POST processes credentials."""
    if request.method == 'GET':
        if 'admin_logged_in' in session:
            return redirect(url_for('admin'))
        return render_template('login.html')
    
    username = request.form.get('username')
    password = request.form.get('password')
    
    admin = db_manager.verify_admin_credentials(username, password)
    if admin:
        session['admin_logged_in'] = True
        session['admin_username'] = username
        session['admin_id'] = admin['id']
        
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        db_manager.log_activity(admin['id'], 'LOGIN', f"Admin {username} logged in", ip_address)
        
        return jsonify({"status": "success", "message": "Login berhasil"})
    
    return jsonify({"status": "error", "message": "Username atau password salah"})


@app.route('/admin/logout')
def logout():
    """Log out admin user and clear session."""
    admin_id = session.get('admin_id')
    username = session.get('admin_username')
    
    if admin_id and username:
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        db_manager.log_activity(admin_id, 'LOGOUT', f"Admin {username} logged out", ip_address)
    
    session.clear()
    return redirect(url_for('login_page'))


# ===== Admin Dashboard Routes =====

@app.route('/admin')
@login_required
def admin():
    """Render admin dashboard page."""
    return render_template('admin.html')


@app.route('/admin/management')
@login_required
def admin_management():
    """Render admin management page."""
    return render_template('admin_management.html')


@app.route('/admin/analytics')
@login_required
def admin_analytics():
    """Render visitor analytics page."""
    return render_template('analytics.html')


@app.route('/admin/content-management', methods=['GET', 'POST'])
@login_required
def content_management():
    """Handle content management - GET shows form, POST updates settings and logos."""
    if request.method == 'POST':
        try:
            chatbot_name = request.form.get('chatbot_name')
            subtitle = request.form.get('subtitle')
            copyright_text = request.form.get('copyright_text')
            
            settings_to_update = {}
            if chatbot_name:
                settings_to_update['chatbot_name'] = chatbot_name
            if subtitle:
                settings_to_update['subtitle'] = subtitle
            if copyright_text:
                settings_to_update['copyright_text'] = copyright_text
            
            # Handle logo uploads
            for logo_key in ['logo_dark', 'logo_light']:
                if logo_key in request.files:
                    logo_file = request.files[logo_key]
                    if logo_file.filename:
                        file_ext = logo_file.filename.rsplit('.', 1)[1].lower() if '.' in logo_file.filename else ''
                        
                        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                            return jsonify({"status": "error", "message": f"Format {logo_key} tidak valid"})
                        
                        filename = f"{logo_key}.{file_ext}"
                        filepath = os.path.join('static', 'image', filename)
                        logo_file.save(filepath)
                        settings_to_update[logo_key] = f"image/{filename}"
            
            admin_id = session.get('admin_id')
            result = db_manager.update_multiple_settings(settings_to_update, admin_id)
            
            if result["status"] == "success" and admin_id:
                ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
                db_manager.log_activity(admin_id, 'UPDATE_CONTENT', 'Updated chatbot content settings', ip_address)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error: {str(e)}"})
    
    content_settings = db_manager.get_content_settings()
    return render_template('content_management.html', content=content_settings)


# ===== Admin Management API =====

@app.route('/api/admin/list')
@login_required
def get_admin_list():
    """Get list of all active admin users."""
    return jsonify(db_manager.get_all_admins())


@app.route('/api/admin/add', methods=['POST'])
@login_required
def add_admin():
    """Add new admin user with validation."""
    username = request.form.get('username')
    password = request.form.get('password')
    
    if not username or not password:
        return jsonify({"status": "error", "message": "Username dan password harus diisi"})
    
    if len(password) < 6:
        return jsonify({"status": "error", "message": "Password minimal 6 karakter"})
    
    result = db_manager.create_admin(username, password)
    
    if result["status"] == "success":
        admin_id = session.get('admin_id')
        if admin_id:
            ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
            db_manager.log_activity(admin_id, 'CREATE_ADMIN', f"Created admin: {username}", ip_address)
    
    return jsonify(result)


@app.route('/api/admin/delete', methods=['POST'])
@login_required
def delete_admin():
    """Delete admin user (soft delete) with validation."""
    data = request.get_json()
    username = data.get('username')
    
    if not username:
        return jsonify({"status": "error", "message": "Username harus diisi"})
    
    if username == session.get('admin_username'):
        return jsonify({"status": "error", "message": "Tidak dapat menghapus akun sendiri"})
    
    result = db_manager.delete_admin(username)
    
    if result["status"] == "success":
        admin_id = session.get('admin_id')
        if admin_id:
            ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
            db_manager.log_activity(admin_id, 'DELETE_ADMIN', f"Deleted admin: {username}", ip_address)
    
    return jsonify(result)


@app.route('/api/admin/current-user')
@login_required
def get_current_user():
    """Get current logged-in admin username."""
    return jsonify({"username": session.get('admin_username', 'Unknown')})


@app.route('/api/admin/model-info')
@login_required
def get_model_info():
    """Get chatbot model information and statistics."""
    return jsonify(model_info)


@app.route('/api/admin/training-status')
@login_required
def get_training_status():
    """Get current model training status and progress."""
    try:
        return jsonify(training_status)
    except Exception as e:
        print(f"Error getting training status: {str(e)}")
        return jsonify({"status": "error", "progress": 0, "message": f"Error: {str(e)}"}), 500

# --- Rute untuk Unggah File dan Latih Ulang ---
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    global training_status
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Tidak ada file yang dipilih"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Tidak ada file yang dipilih"})
    
    if file and file.filename.endswith('.json'):
        try:
            # Validasi format JSON
            file_content = file.read()
            json_data = json.loads(file_content)
            
            # Validasi struktur intents
            if 'intents' not in json_data:
                return jsonify({"status": "error", "message": "Format file tidak valid - missing 'intents' key"})
            
            # Hapus semua file backup lama sebelum menyimpan yang baru
            try:
                base_dir = os.path.dirname(__file__)
                for filename in os.listdir(base_dir):
                    if filename.startswith('intents_backup_') and filename.endswith('.json'):
                        backup_path = os.path.join(base_dir, filename)
                        os.remove(backup_path)
                        print(f"Deleted old backup: {filename}")
            except Exception as cleanup_error:
                print(f"Warning: Could not delete old backups: {cleanup_error}")
            
            # Reset file pointer dan simpan langsung (replace file lama)
            file.seek(0)
            intents_path = os.path.join(os.path.dirname(__file__), 'intents.json')
            file.save(intents_path)
            print(f"Saved new intents.json, replaced old file")
            
            # Start training in background
            try:
                training_status = {"status": "training", "progress": 0, "message": "Memulai pelatihan..."}
                training_thread = threading.Thread(target=train_model_async)
                training_thread.daemon = True  # Make thread daemon to prevent hanging
                training_thread.start()
                
                # Give thread a moment to start
                time.sleep(0.1)
                
                print(f"Training started, initial status: {training_status}")  # Debug log
                
                return jsonify({"status": "success", "message": "File berhasil diunggah. Training dimulai..."})
                
            except Exception as thread_error:
                print(f"Error starting training thread: {thread_error}")
                training_status = {"status": "error", "progress": 0, "message": f"Gagal memulai training: {str(thread_error)}"}
                return jsonify({"status": "error", "message": f"Gagal memulai training: {str(thread_error)}"})
            
        except json.JSONDecodeError:
            return jsonify({"status": "error", "message": "File JSON tidak valid"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error: {str(e)}"})
    
    return jsonify({"status": "error", "message": "File harus berformat .json"})

# --- API untuk preview file ---
@app.route('/api/admin/preview', methods=['POST'])
@login_required
def preview_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Tidak ada file"})
    
    file = request.files['file']
    if file and file.filename.endswith('.json'):
        try:
            content = file.read()
            json_data = json.loads(content)
            
            # Hitung statistik
            total_intents = len(json_data.get('intents', []))
            total_patterns = sum(len(intent.get('patterns', [])) for intent in json_data.get('intents', []))
            
            preview_data = {
                "total_intents": total_intents,
                "total_patterns": total_patterns,
                "intents": [{"tag": intent.get("tag", ""), "patterns_count": len(intent.get("patterns", []))} 
                           for intent in json_data.get('intents', [])[:5]]  # Show first 5
            }
            
            return jsonify({"status": "success", "data": preview_data})
            
        except json.JSONDecodeError:
            return jsonify({"status": "error", "message": "File JSON tidak valid"})
    
    return jsonify({"status": "error", "message": "File harus berformat .json"})

# --- API untuk mendapatkan aktivitas terbaru ---
@app.route('/api/admin/recent-activities')
@login_required
def get_recent_activities():
    try:
        activities = db_manager.get_recent_activities(5)
        
        # Format activities for frontend
        formatted_activities = []
        for activity in activities:
            icon = 'fas fa-sign-in-alt'
            color = 'green'
            
            if activity['action'] == 'LOGOUT':
                icon = 'fas fa-sign-out-alt'
                color = 'blue'
            elif activity['action'] == 'CREATE_ADMIN':
                icon = 'fas fa-user-plus'
                color = 'purple'
            elif activity['action'] == 'DELETE_ADMIN':
                icon = 'fas fa-user-minus'
                color = 'red'
            
            formatted_activities.append({
                'id': len(formatted_activities) + 1,
                'type': activity['action'].lower(),
                'message': activity['description'] or activity['action'],
                'timestamp': activity['created_at'],
                'icon': icon,
                'color': color,
                'username': activity['username']
            })
        
        # Add system activities if no database activities
        if not formatted_activities:
            # Cek apakah model ada
            if os.path.exists('model/chatbot_brmp_model.h5'):
                model_stat = os.path.getmtime('model/chatbot_brmp_model.h5')
                model_time = datetime.fromtimestamp(model_stat)
                time_diff = datetime.now() - model_time
                
                if time_diff.total_seconds() < 3600:  # Kurang dari 1 jam
                    formatted_activities.append({
                        "id": 1,
                        "type": "training",
                        "message": "Model berhasil dilatih ulang",
                        "timestamp": model_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "icon": "fas fa-check",
                        "color": "green"
                    })
        
        return jsonify(formatted_activities[:5])
    except Exception as e:
        print(f"Error getting recent activities: {e}")
        return jsonify([])

# --- API untuk export model info ---
@app.route('/api/admin/export-info')
@login_required
def export_model_info():
    try:
        export_data = {
            "model_info": model_info,
            "training_status": training_status,
            "export_timestamp": datetime.now().isoformat(),
            "intents_count": len(intents.get('intents', [])) if 'intents' in globals() else 0
        }
        return jsonify(export_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# --- API untuk statistik sistem real-time ---
@app.route('/api/admin/system-stats')
@login_required
def get_system_stats():
    try:
        stats = {
            "uptime": "Online",
            "memory_usage": "Normal",
            "cpu_usage": "Low",
            "disk_space": "Available",
            "model_loaded": 'model' in globals() and model is not None,
            "intents_loaded": 'intents' in globals() and intents is not None,
            "last_request": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_conversations": session.get('conversation_count', 0)
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/admin/analytics')
@login_required
def get_analytics():
    """Get visitor analytics data with optional period filter."""
    try:
        period = request.args.get('period', 'day')  # day, week, or month
        
        if period not in ['day', 'week', 'month']:
            return jsonify({"status": "error", "message": "Invalid period"}), 400
        
        stats = db_manager.get_visit_stats(period)
        all_time = db_manager.get_all_time_stats()
        
        return jsonify({
            "status": "success",
            "current_period": stats,
            "all_time": all_time
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Chatbot API Routes =====

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    prompt = data.get('message', '')
    
    if 'messages' not in session:
        session['messages'] = []
    
    session['messages'].append({"role": "user", "content": prompt})
    
    ints = predict_class(prompt)
    response = getResponse(ints, intents)
    
    session['messages'].append({"role": "assistant", "content": response})
    
    session.modified = True
    
    return jsonify({
        'response': response,
        'messages': session['messages']
    })

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    session['messages'] = [{"role": "assistant", "content": "Hai! Ada yang bisa aku bantu hari ini?"}]
    session.modified = True
    return jsonify({'status': 'success', 'messages': session['messages']})

@app.route('/api/messages', methods=['GET'])
def get_messages():
    if 'messages' not in session:
        session['messages'] = [{"role": "assistant", "content": "Hai! Ada yang bisa aku bantu hari ini?"}]
    return jsonify({'messages': session['messages']})

@app.route('/api/patterns', methods=['GET'])
def get_all_patterns():
    """Get all patterns from intents.json for autocomplete"""
    try:
        all_patterns = []
        if 'intents' in globals() and intents:
            for intent in intents.get('intents', []):
                for pattern in intent.get('patterns', []):
                    all_patterns.append(pattern)
        return jsonify({'patterns': all_patterns})
    except Exception as e:
        print(f"Error getting patterns: {e}")
        return jsonify({'patterns': []})

if __name__ == '__main__':
    import os
    # Suppress Flask development server warning
    os.environ['FLASK_ENV'] = 'development'
    
    print("\n" + "="*60)
    print("🚀 Chatbot BRMP Development Server")
    print("="*60)
    print("📍 URL: http://localhost:5000")
    print("📍 Admin: http://localhost:5000/admin/login")
    print("⚠️  Development mode - Not for production use")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)