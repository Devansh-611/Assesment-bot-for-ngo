import streamlit as st
from google import genai
from PIL import Image
import json
import re
from dotenv import load_dotenv
import os


# loading the environment variable

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

st.title(" Email Evaluation Tutor & Assesment Bot for Non Profit Organization")



# session init state

if "quiz" not in st.session_state:
    st.session_state.quiz = None
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.show_feedback = False
    st.session_state.answered = False

# uploading the image of email

uploaded_email = st.file_uploader(
    "Upload the screenshot of to Donor Email ",
    type=["png", "jpg", "jpeg" ]
)

num_questions = st.sidebar.slider("Number of Question", 3, 10, 5)


# quiz generation

if uploaded_email and st.button("Generate  Quiz"):

    image = Image.open(uploaded_email)

    with st.spinner("Extracting email text..."):

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                "Extract ONLY the donor email text from this image:",
                image
            ]
        )

    st.session_state.email_text = response.text

    
    
    # generating the questions using prompt
    prompt = f"""
You are a senior nonprofit fundraising communication evaluator.

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

Email:
{st.session_state.email_text}
"""

    with st.spinner("Generating Assesment  questions..."):

        quiz_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

    quiz_text = quiz_response.text.strip()
    quiz_text = re.sub(r"```json|```", "", quiz_text).strip()

    # Safe JSON parsing
    try:
        st.session_state.quiz = json.loads(quiz_text)
    except:
        st.error("AI returned invalid format. Please try again.")
        st.stop()

    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.show_feedback = False
    st.session_state.answered = False

# displaying the email text

if "email_text" in st.session_state:

    with st.expander("Extracted Email Preview"):
        st.write(st.session_state.email_text)

# function to normalise the text to match correct ans and submitted answer
def normalize(text):
    return text.strip().lower()




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

        # Submit Answer
        if st.button("Submit Answer") and not st.session_state.answered:

            st.session_state.show_feedback = True
            st.session_state.answered = True

            selected_index = q["options"].index(answer)

            correct_index = None
            for idx, opt in enumerate(q["options"]):
                if normalize(opt) == normalize(q["correct_answer"]):
                    correct_index = idx
                    break
            
            if selected_index == correct_index:
                st.session_state.score += 1
                st.session_state.feedback = "correct"
            else:
                st.session_state.feedback = "incorrect"




        # Show feedback
        if st.session_state.show_feedback:

            if st.session_state.feedback == "correct":
                st.success("Correct Answer")
            else:
                st.error("Incorrect")

            st.write(" Correct Answer:", q["correct_answer"])
            st.write(" Explanation:", q["explanation"])

            if st.button("Next Question"):

                st.session_state.current_q += 1
                st.session_state.show_feedback = False
                st.session_state.answered = False
                st.rerun()

    else:

        total = len(quiz)
        score = st.session_state.score

        st.header("Final Evaluation")

        st.write(f"Score: {score}/{total}")

        percentage = (score / total) * 100

        if percentage >= 80:
            st.success(" Excellent understanding.")
        elif percentage >= 50:
            st.info(" Good but needs improvement.")
        else:
            st.warning(" Needs improvement — review communication strategy.")
