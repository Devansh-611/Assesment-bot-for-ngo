import streamlit as st
from google import genai
from PIL import Image
import json
import re
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# loading the environment variable and api key

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

# streanlit title

st.title("Email Evaluation Tutor & Assessment Bot for Non-Profit Organizations")


#  settting vector db
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("ngo_emails")

# session state

if "quiz" not in st.session_state:
    st.session_state.quiz = None
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.show_feedback = False
    st.session_state.answered = False

#  uploading multiple images

uploaded_emails = st.file_uploader(
    "Upload donor email screenshots (multiple allowed)",
    type=["png","jpg","jpeg"],
    accept_multiple_files=True
)

# question  no side bar

num_questions = st.sidebar.slider("Number of Questions",3,10,5)

# inserting emails in vector db

if uploaded_emails and st.button("Process Emails & Generate Quiz"):

    extracted_texts = []

    for file in uploaded_emails:

        image = Image.open(file)

        with st.spinner(f"Extracting text from {file.name}..."):

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    "Extract ONLY the donor email text from this image:",
                    image
                ]
            )

        email_text = response.text
        extracted_texts.append(email_text)

        # Adding combined emails to vector DB
        embedding = embedding_model.encode([email_text])[0].tolist()

        collection.add(
            documents=[email_text],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())]
        )

    # Combine extracted texts for preview
    st.session_state.email_text = "\n\n".join(extracted_texts)

#  retriving content fron chorama db or vector db

    query_embedding = embedding_model.encode(
        [st.session_state.email_text]
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    retrieved_context = " ".join(results["documents"][0])

# generating the questions from retrived content

    prompt = f"""
You are a senior nonprofit fundraising communication evaluator.

Use these expert donor email examples:

{retrieved_context}

Generate {num_questions} deep evaluation MCQs.

Focus on:
- persuasion
- emotional storytelling
- donor psychology
- CTA effectiveness
- ethical messaging
- weaknesses

Return ONLY JSON:

[
 {{
  "question":"",
  "options":["","","",""],
  "correct_answer":"",
  "explanation":""
 }}
]
"""

    with st.spinner("Generating assessment questions..."):

        quiz_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

    quiz_text = quiz_response.text.strip()
    quiz_text = re.sub(r"```json|```","",quiz_text).strip()

    try:
        st.session_state.quiz = json.loads(quiz_text)
    except:
        st.error("AI returned invalid JSON format.")
        st.stop()

    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.show_feedback = False
    st.session_state.answered = False

# diaplaying the combined emails text

if "email_text" in st.session_state:

    with st.expander("Extracted Email Preview"):
        st.write(st.session_state.email_text)

#  funtion to normalise the text for correct matching

def normalize(text):
    return text.strip().lower()

# generating quiz

if st.session_state.quiz:

    quiz = st.session_state.quiz
    i = st.session_state.current_q

    if i < len(quiz):

        q = quiz[i]

        st.subheader(f"Question {i+1}")

        answer = st.radio(
            q["question"],
            q["options"],
            key=f"radio_{i}"
        )

        if st.button("Submit Answer") and not st.session_state.answered:

            st.session_state.show_feedback = True
            st.session_state.answered = True

            selected_index = q["options"].index(answer)

            correct_index = None
            for idx,opt in enumerate(q["options"]):
                if normalize(opt)==normalize(q["correct_answer"]):
                    correct_index = idx
                    break

            if selected_index == correct_index:
                st.session_state.score += 1
                st.session_state.feedback="correct"
            else:
                st.session_state.feedback="incorrect"

        if st.session_state.show_feedback:

            if st.session_state.feedback=="correct":
                st.success("Correct Answer")
            else:
                st.error("Incorrect")

            st.write("Correct Answer:",q["correct_answer"])
            st.write("Explanation:",q["explanation"])

            if st.button("Next Question"):

                st.session_state.current_q += 1
                st.session_state.show_feedback=False
                st.session_state.answered=False
                st.rerun()

    else:

        total=len(quiz)
        score=st.session_state.score

        st.header("Final Evaluation")

        st.write(f"Score: {score}/{total}")

        percentage=(score/total)*100

        if percentage>=80:
            st.success("Excellent understanding.")
        elif percentage>=50:
            st.info("Good but needs improvement.")
        else:
            st.warning("Needs improvement — review communication strategy.")
