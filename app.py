# app.py
"""
Adarsh's AI Chatbot using Streamlit + LangChain 0.3+ + Groq
Compatible with Python 3.13 and RunnableWithMessageHistory
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Try importing latest LangChain + Groq
try:
    from langchain_groq import ChatGroq
    from langchain_core.runnables.history import RunnableWithMessageHistory
except ImportError as e:
    st.error(f"Missing dependencies: {e}. Please install latest langchain and langchain-groq")
    st.stop()

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
print(os.getenv("GROQ_API_KEY"))
if not api_key:
    st.error("API key not configured. Get a new Groq API key and set it in .env or Streamlit secrets")
    st.stop()

# ------------------------------
# Streamlit UI setup
# ------------------------------
st.set_page_config(page_title="Adarsh's AI Chatbot", page_icon="🤖")
st.title("Adarsh's AI Chatbot")
st.write("Chat instantly with Groq AI using RunnableWithMessageHistory!")

# Initialize session state for UI messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------
# Display previous chat messages
# ------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------
# Initialize Groq model
# ------------------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",  # latest model
    groq_api_key=api_key,
    temperature=0.7,
)

# ------------------------------
# Chat history classes
# ------------------------------
class ChatMessageHistory:
    """In-memory chat history compatible with RunnableWithMessageHistory"""
    def __init__(self):
        self.messages = []

    def add_user_message(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_ai_message(self, text):
        self.messages.append({"role": "assistant", "content": text})

    def get_messages(self):
        return self.messages

# Session-level storage for multiple conversations
history_store = {}

def get_session_history(session_id: str):
    """Return ChatMessageHistory object for a given session"""
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

# ------------------------------
# Initialize RunnableWithMessageHistory
# ------------------------------
conversation = RunnableWithMessageHistory(
    runnable=model,
    get_session_history=get_session_history,
)

# ------------------------------
# Chat input handling
# ------------------------------
if user_input := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    session_id = st.session_state.get("session_id", "session-1")

    hist = get_session_history(session_id)
    hist.add_user_message(user_input)

    with st.spinner("Thinking..."):
        try:
            # Call the conversation runnable
            response = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            # Extract text content
            assistant_text = getattr(response, "content", None) or str(response)
            hist.add_ai_message(assistant_text)

        except Exception as e:
            assistant_text = f"Error: {e}"

    # Append assistant reply to session_state for UI
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    # Redisplay updated chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
