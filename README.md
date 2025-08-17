# 🚇 DMRC Assistant

A smart chatbot that answers Delhi Metro queries using Gemini AI and a semantic FAQ search system.

## 🎥 Demo

Watch the DMRC Chatbot in action: [Demo Video](https://drive.google.com/file/d/18yudMKlaaqqnDFLzOuxcVqtZwS_Ox-il/view?usp=sharing)

> 💡 Perfect for commuters and tourists seeking Delhi Metro information without queues!

---

## ⚙️ Features 

- **Intelligent Query Processing**: ML models understand DMRC questions
- **Context-Aware Responses**: Retrieves relevant DMRC FAQs
- **Conversation Memory**: Remembers chat history for better context
- **Avatar Customization**: 30+ unique avatars with one-click selection
- **Seamless User Experience**: Intuitive chat interface with modern design

---

## 🛠️ Tech Stack

- **Python 3.8+** — The backbone of the entire project
- **Google Gemini AI** — Advanced language model for intelligent responses
- **Sentence Transformers** — Semantic search and text embeddings
- **Streamlit** — Interactive web interface for seamless user experience
- **FastAPI** — High-performance API backend with automatic documentation
- **Uvicorn** — Lightning-fast ASGI server for production deployment

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Gemini API key

### Installation
```bash
# Clone the repository
git clone [<repository-url>](https://github.com/Luna-3012/DMRC-Chatbot.git)
cd DMRC-Assistant

# Create and activate virtual environment
python -m venv dmrc-venv
source dmrc-venv/bin/activate  # On Windows: dmrc-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"  # On Windows: set GEMINI_API_KEY=your-api-key-here

# Build the vector store
python scripts/embed_and_index.py
```

### Running the Application

##### Option 1: Streamlit Only (Local Mode)
```bash
streamlit run Home.py
```

##### Option 2: Full Stack (Recommended)
```bash
# Terminal 1: Start FastAPI backend
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run Home.py
```

##### API Endpoints
- **Health Check**: http://127.0.0.1:8000/health
- **API Documentation**: http://127.0.0.1:8000/docs
- **Chat Endpoint**: POST http://127.0.0.1:8000/chat

---

## 🔮 Future Scope

- **Multilingual Support** — Hindi, Punjabi, and regional languages
- **Real-time Train Status** — Live updates on train arrivals and delays
- **Voice Input** — Hands-free voice assistance

---

## 💬 Got Ideas? Questions?
I'd love to hear your feedback!
Whether it's a bug, feature suggestion, or just a "hey, this is cool!" — feel free to open an issue or connect with me directly. 

---
