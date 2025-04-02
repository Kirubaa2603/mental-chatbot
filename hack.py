import streamlit as st
import openai
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS  # Using FAISS instead of Hugging Face
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

# Set up OpenAI API key (Replace 'your-api-key' with an actual key)
openai.api_key = "your-api-key"

def initialize_llm():
    return OpenAI(temperature=0.7)

def create_vector_db():
    # Using FAISS as an alternative to Hugging Face embeddings
    sample_texts = ["MindEase is here to help you.", "Take deep breaths and relax."]
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(sample_texts, embeddings)
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """
    You are a compassionate mental health chatbot. Respond thoughtfully:
    {context}
    User: {question}
    Chatbot:
    """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

# Initialize Components
st.set_page_config(page_title="MindEase", layout="wide")
llm = initialize_llm()
vector_db = create_vector_db()
qa_chain = setup_qa_chain(vector_db, llm)

# Sidebar - MindEase Tools
st.sidebar.title("🧘 MindEase Tools")
option = st.sidebar.radio("Choose a tool:", ["💪 Motivation", "📚 Study Tips", "💖 Self-Care"])

if option == "💪 Motivation":
    st.sidebar.write("Need a boost?")
    if st.sidebar.button("✨ Inspire Me!"):
        st.sidebar.success("Believe in yourself! Every step forward is progress.")
    st.sidebar.write("Feeling anxious?")
    if st.sidebar.button("🧘 Anxiety Relief"):
        st.sidebar.success("Take slow, deep breaths. Focus on the present.")

elif option == "📚 Study Tips":
    if st.sidebar.button("📖 Get Study Tips"):
        st.sidebar.success("Try the Feynman Technique: Explain concepts in simple terms!")

elif option == "💖 Self-Care":
    if st.sidebar.button("🛀 Self-Care Reminder"):
        st.sidebar.success("Your shoulders might be tense. Take a stretch break!")

# Main Section
st.title("MindEase: Your Mental Wellness Companion")

# Dropdown - Emotion Selection
emotion = st.selectbox("How do you feel today?", ["😊 Happy", "😢 Sad", "😠 Frustrated", "😌 Calm"])
if emotion == "😊 Happy":
    st.success("That's great! Keep spreading positivity! ✨")
elif emotion == "😢 Sad":
    st.warning("It's okay to feel sad sometimes. You're not alone. 💜")
elif emotion == "😠 Frustrated":
    st.info("Take a deep breath. Let's find a way to ease your frustration.")
elif emotion == "😌 Calm":
    st.success("Enjoy the peace! 🌿")

# Chatbox - Interactive Conversations
st.subheader("I am all ears 👂")
user_input = st.text_input("Talk to me:")
if user_input:
    response = qa_chain.run(user_input)
    st.write(response)

# Study Planner Generator
st.subheader("📅 Study Planner Generator")
num_subjects = st.number_input("How many subjects?", min_value=1, max_value=10, step=1)
for i in range(num_subjects):
    st.text_input(f"Subject {i+1}")
    st.number_input(f"Lessons in Subject {i+1}", min_value=1, step=1)
    st.number_input(f"Time Duration (hrs) for Subject {i+1}", min_value=1, step=1)
if st.button("Generate Study Plan"):
    st.success("Here’s your study plan! 🎯 Stay consistent!")

# Daily Affirmations
st.subheader("🌟 Daily Affirmation")
st.write("You are capable of amazing things. Believe in yourself! 💪")

# Study Timer
st.subheader("⏳ Study Timer")
time_duration = st.slider("Set Study Duration (minutes)", 10, 120, 30)
break_duration = st.radio("Break Time?", ["5 min", "10 min"])
st.success(f"Focus for {time_duration} minutes, then take a {break_duration} break!")
