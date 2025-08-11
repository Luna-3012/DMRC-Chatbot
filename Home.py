import streamlit as st
import requests
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.retriever import retrieve_top_k
from utils.intent_filter import is_dmrc_query
from utils.config import load_config
from utils.session_memory import session_memory
from utils.metro_prompts import get_metro_prompt
from google import genai  
# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="DMRC Chatbot",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Avatar configuration
default_user_avatar = "assets/avatar7.png"
default_bot_avatar = "assets/bot_avatar.png"

user_avatar = st.session_state.get("user_avatar", default_user_avatar)
bot_avatar = default_bot_avatar 

# Display user avatar in sidebar
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.image(user_avatar, width=80, caption="Your Avatar")

# Global button styling
st.markdown("""
<style>
.stButton > button {
    width: 100%;
    background-color: #1f77b4;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 16px;
    font-size: 14px;
    cursor: pointer;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #1565c0;
}
</style>
""", unsafe_allow_html=True)

# Avatar selection button
if st.sidebar.button("👥 Change Avatar", key="change_avatar_btn"):
    avatar_page = Path("pages/Avatar.py")
    if avatar_page.exists():
        st.switch_page(str(avatar_page))
    else:
        st.error("Avatar page not found. Please ensure `pages/Avatar.py` exists.")

# Initialize Gemini AI configuration and API settings
try:
    config = load_config()
    api_key = os.getenv(config["llm"]["api_key_env"])
    client = None
    model_name = config["llm"]["model"]
    if api_key:
        client = genai.Client(api_key=api_key)
        st.session_state.api_mode = True
    else:
        st.warning("⚠️ Gemini API key not found. Set GEMINI_API_KEY environment variable.")
except Exception as e:
    st.error(f"❌ Configuration error: {e}")

# API config
API_ENABLED = False
API_BASE_URL = None
API_TIMEOUT = 8
try:
    api_cfg = config.get("api", {})
    API_ENABLED = bool(api_cfg.get("enabled", False))
    API_BASE_URL = api_cfg.get("base_url", "http://127.0.0.1:8000")
    API_TIMEOUT = int(api_cfg.get("timeout_s", 8))
except Exception:
    API_ENABLED = False
    API_BASE_URL = None
    API_TIMEOUT = 8

# Initialize session state for chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_mode" not in st.session_state:
    st.session_state.api_mode = False

# Generate unique session ID for conversation tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize conversation memory
if not session_memory.get_session_info(st.session_state.session_id):
    session_memory.create_session(st.session_state.session_id)

def build_contextual_prompt(user_query, top_k_context):
    """Build prompt with conversation memory using SessionMemory."""
    session_id = st.session_state.session_id
    
    # Get conversation context from session memory
    conversation_context = session_memory.get_conversation_context(session_id, count=3)
    
    # Build FAQ context
    faq_context = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in top_k_context])
    
    # Combine everything
    final_prompt = (
        f"You are a helpful assistant for the Delhi Metro Rail Corporation (DMRC).\n\n"
        f"{conversation_context}"
        f"Use the following FAQs as context:\n{faq_context}\n\n"
        f"Now answer the user's question clearly and conversationally, "
        f"considering the conversation history:\n{user_query}\n\nAnswer:"
    )
    
    return final_prompt

# Sidebar
with st.sidebar:
    st.title("🚇 DMRC Chatbot")
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.markdown("""
    DMRC Assistant is your smart helpdesk for all things Delhi Metro. ✨
    
    Have a question? Just ask — the assistant understands queries like:
    "How do I get a metro card?"
    "Can I carry luggage?"
    "Where is the Metro Museum?"
                
    No waiting in line. Just ask — and go!!! 💨              
    """)
    
    st.markdown("---")
    
    # Settings Button
    if st.button("⚙️ Settings"):
        st.session_state.show_settings = not st.session_state.get('show_settings', False)
    
    # Settings Panel (only shown when button is clicked)
    if st.session_state.get('show_settings', False):
        st.subheader("🔍 Response Tuning")
        top_k = st.slider("Top K Results", 1, 5, 3)
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.55, 0.05)
        
        # Memory settings
        st.subheader("🧠 Memory Settings")
        memory_enabled = st.checkbox("Enable Conversation Memory", value=True)
        memory_length = st.slider("Memory Length", 1, 10, 5)
    else:
        # Default values when settings are hidden
        top_k = 3
        threshold = 0.55
        memory_enabled = True
        memory_length = 5
    
    # Behind the Scenes Button
    if st.button("🛤️ Behind the Scenes"):
        st.session_state.show_behind_scenes = not st.session_state.get('show_behind_scenes', False)
    
    # Behind the Scenes Panel (only shown when button is clicked)
    if st.session_state.get('show_behind_scenes', False):
        # Session Info
        st.subheader("📊 Session Info")
        session_info = session_memory.get_session_info(st.session_state.session_id)
        if session_info:
            st.write(f"**Session ID:** `{session_info['session_id'][:8]}...`")
            st.write(f"**Conversations:** {session_info['total_conversations']}")
            st.write(f"**Created:** {session_info['created_at'][:19]}")
            st.write(f"**Last Active:** {session_info['last_accessed'][:19]}")
        
        # Memory Stats
        st.subheader("📈 Memory Stats")
        memory_stats = session_memory.get_memory_stats()
        st.write(f"**Active Sessions:** {memory_stats['active_sessions']}")
        st.write(f"**Total Conversations:** {memory_stats['total_conversations']}")
        st.write(f"**Max Sessions:** {memory_stats['max_sessions']}")
        
        # Response Details Toggle
        st.subheader("📋 Response Details")
        show_response_details = st.checkbox("Show Response Details", value=st.session_state.get('show_response_details', False))
        st.session_state.show_response_details = show_response_details
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        session_memory.reset_session(st.session_state.session_id)
        st.rerun()

# Main chat interface
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🚇 DMRC Assistant</h1>', unsafe_allow_html=True)

# Display chat messages with avatars
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(message["content"])
            if "metadata" in message and st.session_state.get('show_response_details', False):
                with st.expander("📊 Response Details"):
                    st.json(message["metadata"])
    else:
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(message["content"])
            if "metadata" in message and st.session_state.get('show_response_details', False):
                with st.expander("📊 Response Details"):
                    st.json(message["metadata"])

# Chat input
if prompt := st.chat_input("Ask me about Delhi Metro services..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar=bot_avatar):
        with st.spinner("💭 Thinking..."):
            try:
                response_data = {"response": "", "source": "", "confidence": 0.0}
                top_k_context = []  
                
                # Try FastAPI first if enabled
                used_api = False
                if API_ENABLED and API_BASE_URL:
                    try:
                        api_payload = {
                            "query": prompt,
                            "session_id": st.session_state.session_id,
                            "top_k": top_k,
                            "threshold": threshold,
                            "memory_enabled": memory_enabled,
                        }
                        r = requests.post(f"{API_BASE_URL}/chat", json=api_payload, timeout=API_TIMEOUT)
                        if r.ok:
                            api_resp = r.json()
                            response_data = {
                                "response": api_resp.get("response", ""),
                                "source": api_resp.get("source", "dmrc_rag"),
                                "confidence": api_resp.get("confidence", 0.8),
                            }
                            ctx = api_resp.get("context", [])
                            top_k_context = [(c.get("question", ""), c.get("answer", "")) for c in ctx]
                            used_api = True
                        else:
                            st.info("Backend API unavailable, using local pipeline.")
                    except Exception:
                        st.info("Backend API call failed, using local pipeline.")

                # Intent classification (local pipeline) 
                if not used_api and config.get("use_intent_classifier", True):
                    try:
                        is_dmrc = is_dmrc_query(prompt)
                        if not is_dmrc:
                            if config.get("fallback_to_llm", True) and st.session_state.api_mode:
                                session_id = st.session_state.session_id
                                conversation_context = session_memory.get_conversation_context(session_id, count=3)
                                
                                metro_prompt = get_metro_prompt(prompt, conversation_context)
                                
                                # Generate metro-themed response
                                response = client.models.generate_content(model=model_name, contents=metro_prompt)
                                response_text = getattr(response, "text", str(response))
                                
                                response_data = {
                                    "response": response_text,
                                    "source": "metro_general",
                                    "confidence": 0.8
                                }
                            else:
                                response_data = {
                                    "response": "This chatbot only handles DMRC-related queries. Please ask about Delhi Metro services.",
                                    "source": "intent_filter",
                                    "confidence": 0.0
                                }
                        else:
                            # Retrieve relevant context
                            top_k_context = retrieve_top_k(
                                prompt, k=top_k, threshold=threshold
                            )
                            
                            if not top_k_context:
                                response_data = {
                                    "response": "I couldn't find specific information about that. Please rephrase or ask about Delhi Metro services.",
                                    "source": "no_matches",
                                    "confidence": 0.0
                                }
                            else:
                                # Build contextual prompt with memory
                                if memory_enabled:
                                    final_prompt = build_contextual_prompt(prompt, top_k_context)
                                else:
                                    context_prompt = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in top_k_context])
                                    final_prompt = (
                                        f"You are a helpful assistant for the Delhi Metro Rail Corporation (DMRC).\n\n"
                                        f"Use the following FAQs as context:\n{context_prompt}\n\n"
                                        f"Now answer the user's question clearly and conversationally:\n{prompt}\nAnswer:"
                                    )
                                
                                if st.session_state.api_mode:
                                    response = client.models.generate_content(model=model_name, contents=final_prompt)
                                    text = getattr(response, "text", str(response))
                                    response_data = {
                                        "response": text,
                                        "source": "dmrc_rag",
                                        "confidence": 0.8
                                    }
                                else:
                                    response_data = {
                                        "response": "API mode not available. Please check your Gemini API key.",
                                        "source": "error",
                                        "confidence": 0.0
                                    }
                    except Exception as e:
                        response_data = {
                            "response": f"Sorry, I encountered an error: {str(e)}",
                            "source": "error",
                            "confidence": 0.0
                        }
                
                # Display response
                st.markdown(response_data["response"])
                
                # Update conversation memory only when using local pipeline
                if memory_enabled and not used_api:
                    session_memory.add_conversation(
                        session_id=st.session_state.session_id,
                        user_query=prompt,
                        bot_response=response_data["response"],
                        source=response_data["source"],
                        confidence=response_data["confidence"],
                        context_used=top_k_context,
                        metadata={
                            "top_k": top_k,
                            "threshold": threshold,
                            "memory_enabled": memory_enabled
                        }
                    )
                
                # Add response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_data["response"],
                    "metadata": {
                        "source": response_data["source"],
                        "confidence": response_data["confidence"],
                        "top_k": top_k,
                        "threshold": threshold,
                        "memory_enabled": memory_enabled,
                        "session_id": st.session_state.session_id[:8],
                        "memory_entries": len(session_memory.get_recent_conversations(st.session_state.session_id))
                    }
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>© DMRC Chatbot · AI by Gemini · Embeddings by BGE</p>
</div>
""", unsafe_allow_html=True) 