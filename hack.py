import os
import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import vectorstores
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import datetime
import random

def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key="YOUR_GROQ_API_KEY",
        model_name="llama-3.3-70b-versatile"
    )

def create_vector_db():
    os.makedirs('datas/', exist_ok=True)
    loader = DirectoryLoader('datas/', glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2')
    vector_db = vectorstores.Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully:
    {context}
    user: {question}
    chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

llm = initialize_llm()
db_path = "./chroma_db"
vector_db = vectorstores.Chroma(persist_directory=db_path, embedding_function=HuggingFaceBgeEmbeddings(model_name='all-mpnet-base-v2'))
qa_chain = setup_qa_chain(vector_db, llm)

# Sidebar UI - MindEase Tools
st.sidebar.title("🌿 MindEase Tools")

# Motivation
with st.sidebar.expander("💪 Motivation"):
    if st.button("Need a Boost? Inspire Me!"):
        motivation_quotes = [
            "Believe in yourself! You are capable of great things.",
            "Every day is a fresh start. Make it count!",
            "You are stronger than you think.",
            "Success is the sum of small efforts, repeated daily.",
        ]
        st.success(random.choice(motivation_quotes))
    if st.button("Feeling Anxious? Anxiety Relief"):
        anxiety_tips = [
            "Take deep breaths and count to ten.",
            "Ground yourself – focus on 5 things you see, 4 you touch, 3 you hear.",
            "Stretch or do light movement to release tension.",
        ]
        st.info(random.choice(anxiety_tips))

# Study Tips
with st.sidebar.expander("📚 Study Tips"):
    if st.button("Get Study Techniques"):
        study_tips = [
            "Try the Feynman Technique: Explain the topic in simple words.",
            "Use Pomodoro technique: 25 min study, 5 min break.",
            "Practice active recall: Test yourself instead of re-reading.",
        ]
        st.info(random.choice(study_tips))

# Self-Care
with st.sidebar.expander("🌸 Self-Care"):
    if st.button("Academic Well-Being Tips"):
        self_care_tips = [
            "Your shoulders might be tense. Take a deep breath and relax.",
            "Hydration is key! Take a sip of water now.",
            "Take a moment to stretch. Your body will thank you!",
        ]
        st.warning(random.choice(self_care_tips))

# Main Page UI
st.title("MindEase - Your AI Study & Mental Wellness Companion 💙")

# Emotion-Based Response
st.subheader("🧠 How do you feel today?")
selected_emotion = st.selectbox("Select your emotion:", ["Happy", "Sad", "Anxious", "Motivated", "Stressed", "Confident"])
response_dict = {
    "Happy": "Keep shining! Happiness is contagious.",
    "Sad": "You're stronger than you think. Keep going!",
    "Anxious": "Breathe in... Breathe out... You got this!",
    "Motivated": "Channel your motivation into action!",
    "Stressed": "Take it one step at a time. You've got this!",
    "Confident": "Confidence looks good on you! Keep moving forward!",
}
st.write(response_dict[selected_emotion])

# Chatbot
st.subheader("🗨️ I am all ears!")
chat_input = st.text_input("Type your message...")
if chat_input:
    chat_response = qa_chain.run(chat_input)
    st.write(f"**MindEase:** {chat_response}")

# Study Planner Generator
st.subheader("📅 Study Planner Generator")
num_subjects = st.selectbox("Number of subjects:", [1, 2, 3, 4, 5])
planner = {}
for i in range(num_subjects):
    subject = st.text_input(f"Subject {i+1}")
    lessons = st.number_input(f"Number of lessons in {subject}", min_value=1, step=1)
    planner[subject] = lessons
study_duration = st.number_input("Total study duration (hours):", min_value=1, step=1)
if st.button("Generate Study Plan"):
    st.success("Here's your study plan!")
    for subject, lessons in planner.items():
        st.write(f"📖 {subject}: {study_duration / lessons:.2f} hours per lesson")

# Daily Affirmation
st.subheader("🌞 Daily Affirmation")
today = datetime.date.today()
daily_quotes = [
    "You are enough, just as you are.",
    "Every step forward is progress, no matter how small.",
    "You deserve success and happiness.",
]
st.write(f"📅 {today}: {random.choice(daily_quotes)}")

# Study Timer
st.subheader("⏳ Study Timer")
study_time = st.number_input("Set study duration (minutes):", min_value=1, step=1)
break_time = st.selectbox("Break duration:", [5, 10])
if st.button("Start Timer"):
    st.success(f"Study for {study_time} minutes, then take a {break_time}-minute break!")

st.write("💡 MindEase is here to support you through every step of your journey!")
