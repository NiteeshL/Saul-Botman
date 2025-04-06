import streamlit as st
import os
import time
import base64
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Custom color scheme and styling
custom_css = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    
    /* Main theme colors */
    :root {
        --primary-color: #f7d31e;
        --secondary-color: #da2225;
        --accent-color: #b06a42;
        --text-color: #ffffff;
        --chat-bg: var(--accent-color);
    }
    
    /* Page styling - removing background color overrides */
    .main {
        color: var(--text-color);
    .stApp {
        color: var(--text-color);
    }

    }

    .stApp {
        color: var(--text-color);
    }
    
    /* Header styling */
    .header-container {
        background: var(--accent-color);
        color: var(--text-color);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-text h1, .header-text p {
        color: var(--text-color) !important;
    }
    
    .header-image {
        flex-shrink: 0;
    }
    
    .header-image img {
        max-width: 300px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    

    
    .dancing-script {
        font-size: 60px;
        font-family: 'Dancing Script', cursive !important;
    }
    
    /* Chat container styling */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message styling */
    .stChatMessage {
        background-color: var(--chat-bg);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        color: var(--text-color);
    }
    
    /* Make message text white */
    .stChatMessage p {
        color: var(--text-color) !important;
    }
    
    /* Input box styling */
    .stTextInput input {
        border-radius: 25px;
        border: 2px solid var(--accent-color);
        padding: 10px 20px;
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    /* Button styling */
    div.stButton > button:first-child {
        background-color: var(--secondary-color);
        color: var(--text-color);
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: var(--accent-color);
        transform: translateY(-2px);
    }
    
    /* Clear Chat button specific styling */
    .clear-chat-button {
        background-color: #f0f0f0 !important;
        color: #666666 !important;
        border-radius: 15px !important;
        padding: 0.3rem 1rem !important;
        font-size: 0.8rem !important;
        border: 1px solid #dddddd !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease !important;
    }
    
    .clear-chat-button:hover {
        background-color: #e0e0e0 !important;
        border-color: #cccccc !important;
        transform: translateY(-1px) !important;
    }
    
    /* Status indicator styling */
    div[data-testid="stStatusWidget"] {
        background-color: var(--secondary-color);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Hide default elements */
    #MainMenu, footer, .stDeployButton, #stDecoration {display: none;}
    button[title="View fullscreen"] {display: none;}
</style>
"""

# Page configuration
st.set_page_config(
    page_title="Saul Botman",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Header section
try:
    image_path = "saul.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode()
            
        st.markdown(f"""
            <div class="header-container">
                <div class="header-text">
                    <h1 style="color: white; font-size: 2.5rem;">
                        <span class="dancing-script">Better Call Bot!</span>
                    </h1>
                    <p style="color: #e0e0e0; font-size: 1.2rem;">Did you know that you have rights? The Constitution says you do. And so do I.</p>
                </div>
                <div class="header-image">
                    <img src="data:image/jpeg;base64,{encoded_image}" alt="Saul Goodman">
                </div>
            </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Unable to load image: {e}")

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the prompt template
prompt_template = """
<s>[INST]This is a chat template and As a legal chat bot , your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# Set up the QA chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages with improved styling
for message in st.session_state.messages:
    with st.chat_message(
        message.get("role"),
        avatar="👤" if message.get("role") == "user" else "⚖️"
    ):
        st.write(message.get("content"))

# Chat input with custom styling
input_prompt = st.chat_input("Ask your legal question...")

if input_prompt:
    with st.chat_message("user", avatar="👤"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant", avatar="⚖️"):
        with st.status("Analyzing your question...", expanded=True):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")
            
        col1, col2, col3 = st.columns([4, 1, 4])
        with col2:
            st.button('🗑️ Clear Chat', on_click=reset_conversation, key="clear_chat", help="Clear the conversation history", type="secondary", use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

st.markdown('</div>', unsafe_allow_html=True)