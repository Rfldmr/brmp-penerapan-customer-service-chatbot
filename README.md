# BRMP Penerapan Customer Servoce Chatbot

BRMP Penerapan Public Service Chatbot I developed an intelligent, rule-based chatbot designed to streamline customer service for BRMP Penerapan, a government agency. Built using Python, TensorFlow, Keras, and NLTK, the NLP model achieved a 95.2% training accuracy in understanding and responding to user inquiries. The web application, powered by Flask and Tailwind CSS, features a dual-interface architecture: a responsive public chat interface for users and a comprehensive Admin Dashboard. The secure admin panel includes modules for managing JSON datasets, controlling admin access, customizing website content (CMS), and visualizing visitor statistics, allowing non-technical staff to easily maintain the system.

## 🚀 Quick Start

### Development Mode (Testing Lokal)
```powershell
# Jalankan aplikasi
python app.py

# Akses aplikasi
# Chatbot: http://localhost:5000
# Admin: http://localhost:5000/admin/login
```

## ⚙️ Fitur

- 🤖 AI Chatbot dengan TensorFlow
- 👨‍💼 Admin Dashboard
- 📊 Statistics & Analytics
- 🔐 Authentication System
- 🔄 Auto-refresh Dashboard
- ✨ Fuzzy Matching (Typo Tolerance)
- 📝 Rich Text Formatting
- 🔍 Autocomplete/Suggestion Input
  - User hanya bisa bertanya sesuai patterns di intents.json
  - Real-time suggestions saat mengetik
  - Guided conversation experience

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **AI/ML:** TensorFlow, Keras, NLTK
- **Database:** SQLite3
- **Frontend:** HTML, CSS (Tailwind), JavaScript
- **Production:** Waitress WSGI Server



