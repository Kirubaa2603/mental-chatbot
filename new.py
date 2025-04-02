import streamlit as st
import random
import os
import datetime
import sys
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import vectorstores
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Initialize LLM
def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key="YOUR_GROQ_API_KEY",  # Replace with your actual API key
        model_name="llama-3.3-70b-versatile"
    )

# Create or load vector database
def create_or_load_vector_db():
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')
        return vectorstores.Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    os.makedirs('datas', exist_ok=True)
    loader = DirectoryLoader('datas', glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        st.error("No documents found for training chatbot.")
        sys.exit()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')
    vector_db = vectorstores.Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    vector_db.persist()
    return vector_db

# Setup chatbot chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health chatbot.
    Respond thoughtfully to the following questions:
    {context}
    user: {question}
    chatbot: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

# Initialize chatbot
llm = initialize_llm()
vector_db = create_or_load_vector_db()
qa_chain = setup_qa_chain(vector_db, llm)

# Streamlit App Configuration
st.set_page_config(page_title="MindEase", layout="wide")
st.title("🌿 Welcome to MindEase")
st.subheader("Your personal companion for motivation, study tips, and self-care.")

# Sidebar with Quick Tools
st.sidebar.title("MindEase Tools")

def display_random_prompt(category):
    prompts = {
        "motivation": [
            "Believe in yourself! You are capable of amazing things.",
            "Every day is a new beginning. Take a deep breath and start again.",
            "Your potential is limitless. Never stop exploring your capabilities."
        ],
        "anxiety": [
            "Take a deep breath. Inhale for 4 seconds, hold for 4, and exhale for 6.",
            "Listen to calming music or nature sounds to ease your mind.",
            "Step outside and take a short walk to clear your thoughts."
        ],
        "study": [
            "Use the Pomodoro technique – study for 25 mins, take a 5-min break.",
            "Summarize notes in your own words to enhance understanding.",
            "Break large tasks into smaller chunks to avoid feeling overwhelmed."
        ],
        "selfcare": [
            "Take a 5-minute stretch break to ease your muscles.",
            "Get sunlight exposure to boost your mood and energy levels.",
            "Practice gratitude – write down three things you are grateful for."
        ]
    }
    st.sidebar.write(random.choice(prompts[category]))

if st.sidebar.button("Need Motivation?"):
    display_random_prompt("motivation")
if st.sidebar.button("Feeling Anxious?"):
    display_random_prompt("anxiety")
if st.sidebar.button("Study Tips"):
    display_random_prompt("study")
if st.sidebar.button("Self-care Tips"):
    display_random_prompt("selfcare")

# Chatbot Section
st.subheader("💬 Mental Health Chatbot")
user_input = st.text_input("Ask MindEase anything about mental health:")
if st.button("Ask"):  
    if user_input.strip():
        response = qa_chain.run(user_input)
        st.write("🤖 MindEase:", response)
    else:
        st.error("Please enter a valid question.")

# Daily Affirmation
st.subheader("✨ Daily Affirmation")
affirmations = [
    "You are enough just as you are.",
    "Every challenge you overcome makes you stronger.",
    "You are doing your best, and that is enough."
]
current_date = datetime.datetime.now().day
st.write(affirmations[current_date % len(affirmations)])

# Study Planner Generator
st.subheader("📖 Study Planner Generator")
num_subjects = st.number_input("Number of subjects:", min_value=1, max_value=10, step=1)
study_time = st.number_input("Total study time (in minutes):", min_value=30, step=10)
if st.button("Generate Study Plan"):
    plan = {f"Subject {i+1}": f"Study for {round(study_time / num_subjects, 2)} minutes." for i in range(int(num_subjects))}
    st.write(plan)

# Study Timer
st.subheader("⏳ Study Timer")
study_duration = st.number_input("Set your study duration (minutes):", min_value=10, max_value=180, step=5)
break_time = st.selectbox("Break duration:", [5, 10, 15])
if st.button("Start Timer"):
    st.write(f"Study for {study_duration} minutes, then take a {break_time}-minute break.")
