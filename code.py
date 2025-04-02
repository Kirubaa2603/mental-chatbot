import streamlit as st
import random
import os

# Streamlit App Layout
st.set_page_config(page_title="MindEase", layout="wide")
st.title("🌿 Welcome to MindEase")
st.subheader("Your personal companion for motivation, study tips, and self-care.")

st.sidebar.title("MindEase Tools")

# ========= PROMPT COLLECTIONS =========
motivation_prompts = [
    "Believe in yourself! You are capable of great things.",
    "Keep pushing forward! Every step brings you closer to success.",
    "Success is not final, failure is not fatal. It’s the courage to continue that counts.",
    "Your hard work will pay off, stay consistent!"
]

anxiety_relief_prompts = [
    "Take a deep breath. Inhale... Exhale... Everything will be okay.",
    "You are safe. You are in control. You’ve got this!",
    "Close your eyes and imagine a peaceful place. Hold onto that feeling.",
    "Remember, it’s okay to take breaks and rest. You’re doing your best."
]

study_tips = [
    "Use the Pomodoro technique: 25 minutes of focus, 5-minute break.",
    "Rewrite your notes in your own words—it improves retention!",
    "Teach what you’ve learned to someone else. If you can explain it, you understand it.",
    "Practice active recall—test yourself instead of just reading notes."
]

self_care_tips = [
    "Drink enough water and get some fresh air. Your brain needs it!",
    "Listen to your favorite music and relax. You deserve it!",
    "Journaling your thoughts can help clear your mind.",
    "Take a break, stretch, and move around. Your body needs care too."
]

# ========= SIDEBAR BUTTONS =========
if st.sidebar.button("Need a boost? Inspire Me!"):
    st.sidebar.write(random.choice(motivation_prompts))

if st.sidebar.button("Feeling anxious? Anxiety Relief"):
    st.sidebar.write(random.choice(anxiety_relief_prompts))

if st.sidebar.button("Study Tips"):
    st.sidebar.write(random.choice(study_tips))

if st.sidebar.button("Self-care Tips"):
    st.sidebar.write(random.choice(self_care_tips))

# ========= NAVIGATION =========
selected_option = st.sidebar.radio("Navigation", ["Home", "Chatbot"])

if selected_option == "Home":
    st.write("Explore the tools in the sidebar or interact with our chatbot!")
    st.write("🕒 **Focus Timer:** Use the timer below to stay productive.")
    timer_input = st.number_input("Set timer (minutes):", min_value=1, max_value=120, value=25, step=1)
    if st.button("Start Timer"):
        st.write(f"⏳ Timer started for {timer_input} minutes. Stay focused!")

# ========= IMPORTS & LLM SETUP =========
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# ========= API KEY HANDLING =========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Secure API key storage

if not GROQ_API_KEY:
    st.error("🚨 Groq API key is missing! Set it in environment variables.")
    st.stop()

# ========= LLM INITIALIZATION =========
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-verbose"
    )
    return llm

# ========= VECTOR DATABASE =========
db_path = "./chroma_db"

def create_vector_db():
    embeddings = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-base-en')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vector_db

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-base-en')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# ========= CHATBOT LOGIC =========
llm = initialize_llm()

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health chatbot. Respond thoughtfully:
    {context}
    user: {question}
    chatbot: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

qa_chain = setup_qa_chain(vector_db, llm)

# ========= STREAMLIT CHAT INTERFACE =========
st.subheader("💬 Chat with MindEase")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_message = st.text_input("You:", placeholder="Ask me anything!", key="user_input")

def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input", history
    response = qa_chain.run(user_input)
    history.append(("You", user_input))
    history.append(("MindEase", response))
    return response, history

if user_message:
    bot_response, st.session_state["chat_history"] = chatbot_response(user_message, st.session_state["chat_history"])

for sender, message in st.session_state["chat_history"]:
    if sender == "You":
        st.write(f"**{sender}:** {message}")
    else:
        st.write(f"🤖 **{sender}:** {message}")
