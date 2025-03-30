import streamlit as st
try:
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain_groq import ChatGroq
except ImportError as e:
    st.error(f"Missing dependencies: {str(e)}")
    st.stop()

# Rest of your existing code...
import os
from dotenv import load_dotenv

# Load .env file 
load_dotenv()

# Set your API key (choose one method)
"""api_key = os.getenv("GROQ_API_KEY")  # Recommended
# api_key = st.secrets["GROQ_API_KEY"]  # For Streamlit Cloud"""
api_key = (
    os.getenv("GROQ_API_KEY")          # 1. Environment variable
    or st.secrets.get("GROQ_API_KEY")  # 2. Streamlit secrets
    # or "your-key-here"                 # 3. Hardcoded fallback (remove in production)
)

# Personalize the app
st.title("Adarsh's AI Chatbot")  # 👈 Change to your name
st.write("Chat instantly with no API key required!")

# Initialize conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat logic
if api_key:
    model = ChatGroq(
        model_name="llama3-70b-8192",
        groq_api_key=api_key,
        temperature=0.7,
    )
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=model, memory=memory)

    if user_input := st.chat_input("Say something"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = conversation.predict(input=user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Redisplay updated chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
else:
    st.error("API key not configured. Contact the app owner.")